"""
Human Tracker Node - Full owner identification state machine
States: IDLE → SEARCHING → ENROLLING → OWNED → FOLLOWING/GOING_TO/STOPPED → RESET
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Bool, String, Float32
from cv_bridge import CvBridge
import cv2, numpy as np, json, time, threading, math

from .gesture_detector import GestureDetector
from .person_tracker    import PersonTracker, FX, FRAME_W
from .owner_tracking_upgrade import fov_servo, VelocityShaper

# ── PD Controller ──────────────────────────────────────────────────────────
KP_LIN=0.6; KD_LIN=0.06; KP_ANG=0.55; KD_ANG=0.05
DESIRED_DIST=1.3; STOP_DIST=0.8; MAX_LIN=0.5; MAX_ANG=0.9
DEADBAND_DIST=0.18; DEADBAND_BEAR=0.15; THROTTLE_SCALE=0.7
DERIV_CLAMP=2.0

class PDController:
    def __init__(self):
        self._pez=self._pex=0.0; self._pt=time.monotonic()

    def compute(self, z, cx, fw=640):
        now=time.monotonic(); dt=max(0.02,now-self._pt); self._pt=now
        ez=z-DESIRED_DIST; ex=-(cx-fw/2)/(fw/2)  # negated: camera is mirrored
        if z < STOP_DIST:
            self._pez=ez; self._pex=ex
            return 0.0, 0.0
        if abs(ez) < DEADBAND_DIST:
            lin = 0.0
        else:
            dez=(ez-self._pez)/dt; dez=max(-DERIV_CLAMP,min(DERIV_CLAMP,dez))
            lin=max(0.0, KP_LIN*ez+KD_LIN*dez)
        if abs(ex) < DEADBAND_BEAR:
            ang = 0.0
        else:
            dex=(ex-self._pex)/dt; dex=max(-DERIV_CLAMP,min(DERIV_CLAMP,dex))
            ang=-(KP_ANG*ex+KD_ANG*dex)
        self._pez=ez; self._pex=ex
        lin*=THROTTLE_SCALE
        # Ackermann steers only while moving — no servo thrash when stopped
        if lin < 0.02:
            ang = 0.0
        return float(np.clip(lin,-MAX_LIN,MAX_LIN)), float(np.clip(ang,-MAX_ANG,MAX_ANG))

    def reset(self): self._pez=self._pex=0.0; self._pt=time.monotonic()

# ── States ─────────────────────────────────────────────────────────────────
IDLE       = "IDLE"
SEARCHING  = "SEARCHING"
ENROLLING  = "ENROLLING"
OWNED      = "OWNED"
FOLLOWING  = "VOICE_FOLLOW"
GOING_TO   = "VOICE_GOTO"
STOPPED    = "VOICE_STOP"
RESETTING  = "RESETTING"


class HumanTrackerNode(Node):
    def __init__(self):
        super().__init__('human_tracker_node')
        self.bridge   = CvBridge()
        self.gesture  = GestureDetector()
        self.tracker  = PersonTracker()
        # Experimental control (Phase 4). The fast path decides ~66% of
        # frames by Kalman gate + single-cue ReID, bypassing the six-cue
        # fusion. Ablations that evaluate the fusion must disable it.
        self.declare_parameter('fast_path_enabled', True)
        self.tracker.fast_path_enabled = bool(
            self.get_parameter('fast_path_enabled').value)
        self.get_logger().info(
            f'fast_path_enabled={self.tracker.fast_path_enabled}')
        self.pd       = PDController()
        self.shaper   = VelocityShaper(max_accel=0.5, max_ang_accel=1.8)  # [P2] smooth cmd_vel

        self.rgb   = None
        self.depth = None
        self._lock = threading.Lock()
        self.state = IDLE

        # Searching state
        self._search_start    = 0.0
        self._search_timeout  = 30.0   # seconds to search before stopping
        self._search_yaw      = 0.0    # cumulative yaw during search
        self._last_owner_cx   = 320    # last horizontal pos of owner (frame center=320)
        self._lost_search_dir = 0.0    # +left / -right to spin when lost
        self._enroll_person   = None   # detection being enrolled

        # Circular motion state
        self._circ_end  = 0; self._circ_ang = 0.8; self._circ_sign = 1.0

        # Subscriptions
        self.create_subscription(Image,  '/camera/color/image_raw', self._rgb_cb,   10)
        self.create_subscription(Image,  '/camera/depth/image_raw', self._depth_cb, 10)
        self.create_subscription(String, '/voice_command',           self._voice_cb, 10)

        # Click-to-select target (replaces gesture enrollment)
        self.create_subscription(String, '/select_target', self._select_cb, 10)
        # Publishers
        self.ann_pub   = self.create_publisher(Image,       '/tracker/annotated_image', 10)
        self.pose_pub  = self.create_publisher(PoseStamped, '/tracker/target_pose',     10)
        self.owner_dist_pub = self.create_publisher(Float32, '/owner_distance', 10)
        self.diag_pub       = self.create_publisher(String,  '/tracker/diagnostics',    10)
        self.lock_pub  = self.create_publisher(Bool,        '/tracker/target_locked',   10)
        self._loop_count  = 0
        self._loop_rate_t = time.time()
        self._gesture_ms  = 0.0
        self._gesture_tick = 0          # throttle gesture to every Nth frame
        self._last_gd      = None       # cache last gesture result
        self.cmd_pub   = self.create_publisher(Twist,       '/cmd_vel',                 10)
        self.state_pub = self.create_publisher(String,      '/robot_state',             10)
        self.eye_pub   = self.create_publisher(String,      '/eye_state',               10)

        self.create_timer(0.033, self._loop)
        self._pub_state(IDLE)
        self.get_logger().info('Human Tracker Node (YOLO+ReID+OwnerID) initialized')

    # ── Callbacks ──────────────────────────────────────────────────────────
    def _rgb_cb(self, msg):
        try: self.rgb = self.bridge.imgmsg_to_cv2(msg,'bgr8')
        except: pass

    def _depth_cb(self, msg):
        try:
            self.depth = self.bridge.imgmsg_to_cv2(msg,'16UC1')
        except Exception as e:
            print(f'[DEPTH_CB_ERR] {e}')

    def _select_cb(self, msg):
        """Operator clicked a person in the video stream.

        Accepted from any state: re-selecting while already OWNED is how the
        owner is handed over, which the L5 scenario needs. The click is the
        authority — there is no gesture to confirm it against.
        """
        try:
            d = json.loads(msg.data)
            nx, ny = float(d['x']), float(d['y'])
        except Exception as e:
            self.get_logger().error(f'select parse error: {e}')
            return

        frame = self.rgb
        if frame is None:
            self.get_logger().warn('select ignored — no frame yet')
            return

        with self._lock:
            pd = self.tracker.detect_person_at_point(frame.copy(), nx, ny, self.depth)
            if not (pd and pd['person_detected']):
                self.get_logger().warn(f'select missed — no person at ({nx:.2f},{ny:.2f})')
                return
            self._stop()
            self.tracker.reset_owner()
            self._enroll_person = pd
            self.tracker.start_enrollment()
            self._transition(ENROLLING)
            self.get_logger().info(f'Target selected at ({nx:.2f},{ny:.2f}) — enrolling')

    def _voice_cb(self, msg):
        try:
            cmd = json.loads(msg.data)
            self._apply_voice(cmd)
        except Exception as e:
            self.get_logger().error(f'Voice parse error: {e}')

    def _apply_voice(self, cmd):
        action = cmd.get("action","unknown")
        with self._lock:
            # Owner release commands
            if action == "release_owner":
                self._transition(RESETTING)
                return

            # Wake trigger - start searching for owner
            if action == "wake_trigger":
                if self.state == IDLE:
                    self._transition(SEARCHING)
                return

            # Only accept commands if owner is enrolled
            if self.state in (IDLE, SEARCHING, ENROLLING):
                return

            if action == "follow":
                self.tracker.lock_target({'person_detected':True,
                    'position_3d':self.tracker.target_position,
                    'position_2d':(320,240),'bbox':(240,100,160,320)})
                self._transition(FOLLOWING)
                self.pd.reset()
            elif action == "stop":
                self._transition(STOPPED)
                self._stop()
            elif action == "circular":
                self._circ_ang  = float(cmd.get("angular",0.8))
                self._circ_sign = 1.0 if cmd.get("direction","left")=="left" else -1.0
                self._circ_end  = time.time()+float(cmd.get("duration",3))
                self._transition("VOICE_CIRCULAR")
            elif action == "go_to":
                self._transition(GOING_TO)
                self.pd.reset()
            elif action == "idle":
                self._transition(OWNED)
                self._stop()

    # ── Main loop ──────────────────────────────────────────────────────────
    def _loop(self):
        if self.rgb is None: return
        _loop_t0 = time.perf_counter()
        frame = self.rgb.copy()
        depth = self.depth

        with self._lock:
            state = self.state

        if   state == IDLE:       self._do_idle(frame, depth)
        elif state == SEARCHING:  self._do_searching(frame, depth)
        elif state == ENROLLING:  self._do_enrolling(frame, depth)
        elif state == OWNED:      self._do_owned(frame, depth)
        elif state == FOLLOWING:  self._do_following(frame, depth)
        elif state == GOING_TO:   self._do_goto(frame, depth)
        elif state == STOPPED:    self._do_stopped(frame, depth)
        elif state == "VOICE_CIRCULAR": self._do_circular(frame)
        elif state == RESETTING:  self._do_reset(frame)

        self._pub_frame(frame)

        # Phase 2 — per-loop diagnostics
        loop_ms = (time.perf_counter() - _loop_t0) * 1000
        self._loop_count += 1
        now = time.time()
        elapsed = now - self._loop_rate_t
        if elapsed >= 1.0:
            rate_hz = self._loop_count / elapsed
            self._loop_count = 0
            self._loop_rate_t = now
            perf = getattr(self.tracker, "perf", {})
            import json
            diag = json.dumps({
                "rate_hz":    round(rate_hz, 2),
                "loop_ms":    round(loop_ms, 2),
                "yolo_ms":    round(perf.get("yolo_ms", 0), 2),
                "reid_ms":    round(perf.get("reid_ms", 0), 2),
                "fusion_ms":  round(perf.get("fusion_ms", 0), 2),
                "gesture_ms": round(self._gesture_ms, 2),
                "state":      state,
            })
            msg = String(); msg.data = diag
            self.diag_pub.publish(msg)

    # ── State handlers ─────────────────────────────────────────────────────
    def _do_idle(self, frame, depth):
        self._draw(frame, "Say 'Marco' to activate", (128,128,128))

    def _do_searching(self, frame, depth):
        """Rotate and look for a waving person."""
        elapsed = time.time() - self._search_start

        self._gesture_tick += 1
        if self._gesture_tick % 12 == 0 or self._last_gd is None:
            _tg = time.perf_counter()
            gd = self.gesture.detect_gestures(frame)
            self._gesture_ms = (time.perf_counter() - _tg) * 1000
            self._last_gd = gd
        else:
            gd = self._last_gd
        frame[:] = gd['annotated_frame']

        if elapsed > self._search_timeout:
            self.get_logger().warn('Search timeout — returning to IDLE')
            self._transition(IDLE)
            self._stop()
            return

        # Draw status
        self._draw(frame, f"Searching... wave at me! ({int(self._search_timeout-elapsed)}s)",
                   (255,200,0), y=30)
        remaining_arc = int(360 - (elapsed/self._search_timeout)*360)
        self._draw(frame, f"Scanning {remaining_arc}°", (200,200,0), y=60)

        if gd['wave_confirmed'] and gd['wave_bbox']:
            self.get_logger().info('Wave confirmed! Enrolling owner...')
            # Find the person waving
            pd = self.tracker.detect_waving_person(frame, gd['wave_bbox'], depth)
            if pd and pd['person_detected']:
                self._stop()
                self._enroll_person = pd
                self.tracker.start_enrollment()
                self._transition(ENROLLING)
                return

        # Arc manoeuvre — Ackermann needs linear.x to actuate steering (F36)
        twist = Twist()
        twist.linear.x  = 0.12
        twist.angular.z = 0.3
        self.cmd_pub.publish(twist)

    def _do_enrolling(self, frame, depth):
        """Save owner features over multiple frames."""
        pd = self._enroll_person
        if pd is None:
            self._transition(SEARCHING)
            return

        # Re-detect person at same location
        dets = self.tracker._detect(frame)
        if dets:
            if pd.get('raw_bbox'):
                from .person_tracker import iou
                best = max(dets, key=lambda d: iou(pd['raw_bbox'], d))
                x1,y1,x2,y2 = best
                pd['raw_bbox'] = best
                pd['bbox']     = (x1,y1,x2-x1,y2-y1)
                pd['position_3d'] = self.tracker._get_3d(depth, best)

        # Draw enrollment progress
        prog = self.tracker.enroll_frames
        total = self.tracker.ENROLL_FRAMES_NEEDED
        pct  = int(prog/total*100)
        self._draw(frame, f"Enrolling owner... {pct}%", (0,255,200), y=30)
        if pd.get('bbox'):
            x,y,w,h = pd['bbox']
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,200),3)
            cv2.rectangle(frame,(x,y+h+5),(x+int(w*pct/100),y+h+15),(0,255,200),-1)

        self.tracker._cur_depth = depth   # make depth available to enroll_frame
        done = self.tracker.enroll_frame(frame, pd.get('raw_bbox') or
               [pd['bbox'][0], pd['bbox'][1],
                pd['bbox'][0]+pd['bbox'][2], pd['bbox'][1]+pd['bbox'][3]])

        if done:
            self.tracker.lock_target(pd)
            self._transition(OWNED)
            self.get_logger().info('Owner enrolled and locked!')
            msg = Bool(); msg.data=True; self.lock_pub.publish(msg)
            self._pub_eye("following")

    def _do_owned(self, frame, depth):
        """Owner identified — track position, wait for commands."""
        pd = self.tracker.find_owner(frame, depth)
        if pd:
            frame[:] = self.tracker.draw_tracking_info(frame, pd)
            self._pub_pose(pd)
            self._draw(frame,"OWNER IDENTIFIED - awaiting command",(0,255,0),y=30)
        else:
            self._draw(frame,"Owner not visible",(200,200,0),y=30)

        # Check for stop sign (release owner)
        self._gesture_tick += 1
        if self._gesture_tick % 12 == 0 or self._last_gd is None:
            _tg = time.perf_counter()
            gd = self.gesture.detect_gestures(frame)
            self._gesture_ms = (time.perf_counter() - _tg) * 1000
            self._last_gd = gd
        else:
            gd = self._last_gd
        if gd['stop_sign']:
            self._transition(RESETTING)

    def _do_following(self, frame, depth):
        """Follow enrolled owner."""
        pd = self.tracker.find_owner(frame, depth)
        if pd is None:
            # Arc toward last-seen side — Ackermann needs linear.x (F36)
            self._draw(frame,"SEARCHING - spinning to find you",(0,200,200),y=30)
            spin = 0.5 if self._last_owner_cx < 320 else -0.5
            tw = Twist()
            tw.linear.x  = 0.12      # F36: minimum linear for Ackermann steering
            tw.angular.z = spin
            self.cmd_pub.publish(tw)
            return

        frame[:] = self.tracker.draw_tracking_info(frame, pd)
        self._draw(frame,"FOLLOWING OWNER",(0,255,0),y=30)
        self._pub_pose(pd)

        # Check stop sign
        self._gesture_tick += 1
        if self._gesture_tick % 12 == 0 or self._last_gd is None:
            _tg = time.perf_counter()
            gd = self.gesture.detect_gestures(frame)
            self._gesture_ms = (time.perf_counter() - _tg) * 1000
            self._last_gd = gd
        else:
            gd = self._last_gd
        if gd['stop_sign']:
            self._transition(STOPPED); self._stop(); return

        if pd.get('position_3d'):
            z  = pd['position_3d'][2]
            cx = pd['position_2d'][0]
            self._last_owner_cx = cx   # remember side for lost-search spin
            lin,ang = self.pd.compute(z, cx)
            # A coasted (predicted) position is an extrapolation, not a
            # measurement. Driving it at full gain makes the robot chase the
            # filter's guess and then lurch when a real detection arrives.
            # Softening the command while coasting keeps motion continuous
            # without amplifying prediction error.
            if pd.get('predicted'):
                lin *= 0.6; ang *= 0.6
            lin, ang = self.shaper.shape(lin, ang)
            twist=Twist(); twist.linear.x=lin; twist.angular.z=ang
            self.cmd_pub.publish(twist)
        else:
            self._stop()              # F44: owner visible but depth unavailable

    def _do_goto(self, frame, depth):
        """Move toward owner, stop at safe distance."""
        pd = self.tracker.find_owner(frame, depth)
        if pd is None:
            self._draw(frame,"GOTO - searching owner",(0,120,255),y=30)
            twist=Twist()
            twist.linear.x  = 0.12   # F36: Ackermann needs linear to steer
            twist.angular.z = 0.3
            self.cmd_pub.publish(twist)
            return

        frame[:] = self.tracker.draw_tracking_info(frame, pd)
        self._draw(frame,"GOING TO OWNER",(0,120,255),y=30)
        self._pub_pose(pd)

        if pd.get('position_3d'):
            z  = pd['position_3d'][2]
            cx = pd['position_2d'][0]
            if z <= STOP_DIST + 0.1:
                self._stop()
                self._transition(OWNED)
                self.get_logger().info(f'Arrived at owner (dist={z:.2f}m)')
                return
            lin,ang = self.pd.compute(z, cx)
            # fov_servo REMOVED - it pinned steering to max near frame edge
            lin, ang = self.shaper.shape(lin, ang)
            twist=Twist(); twist.linear.x=lin; twist.angular.z=ang
            self.cmd_pub.publish(twist)
        else:
            self._stop()              # F44: owner visible but depth unavailable

    def _do_stopped(self, frame, depth):
        """Stopped — track owner position but don't move."""
        pd = self.tracker.find_owner(frame, depth)
        if pd:
            frame[:] = self.tracker.draw_tracking_info(frame, pd)
            self._pub_pose(pd)
        self._draw(frame,"STOPPED - waiting for command",(0,0,220),y=30)
        self._stop()

        self._gesture_tick += 1
        if self._gesture_tick % 12 == 0 or self._last_gd is None:
            _tg = time.perf_counter()
            gd = self.gesture.detect_gestures(frame)
            self._gesture_ms = (time.perf_counter() - _tg) * 1000
            self._last_gd = gd
        else:
            gd = self._last_gd
        if gd['stop_sign']:
            self._transition(RESETTING)

    def _do_circular(self, frame):
        if time.time() < self._circ_end:
            self._draw(frame,"CIRCULAR",(0,165,255))
            twist=Twist(); twist.linear.x=0.15
            twist.angular.z=self._circ_sign*self._circ_ang
            self.cmd_pub.publish(twist)
        else:
            self._stop(); self._transition(OWNED)

    def _do_reset(self, frame):
        """Release owner and return to IDLE. Non-blocking — returns each tick until 2s elapsed."""
        self._stop()
        if not hasattr(self, '_reset_start'):
            self._reset_start = time.time()
            self.tracker.reset_owner()
            self.gesture.reset()
        self._draw(frame,"Owner released — say Marco to find new owner",(200,0,0),y=30)
        if time.time() - self._reset_start < 2.0:
            return
        del self._reset_start
        self._transition(IDLE)
        msg=Bool(); msg.data=False; self.lock_pub.publish(msg)
        self.get_logger().info('Owner released — returning to IDLE')

    # ── Helpers ────────────────────────────────────────────────────────────
    def _transition(self, new_state):
        old = self.state
        self.state = new_state
        self._pub_state(new_state)
        self.get_logger().info(f'State: {old} -> {new_state}')

        EYE_MAP = {
            IDLE:'idle', SEARCHING:'listening', ENROLLING:'thinking',
            OWNED:'idle', FOLLOWING:'following', STOPPED:'idle',
            GOING_TO:'following', RESETTING:'stopped', "VOICE_CIRCULAR":'circular'
        }
        self._pub_eye(EYE_MAP.get(new_state,'idle'))

        if new_state == SEARCHING:
            self._search_start = time.time()
            self.gesture.reset()

    def _pub_state(self, s):
        msg=String(); msg.data=s; self.state_pub.publish(msg)

    def _pub_eye(self, s):
        msg=String(); msg.data=s; self.eye_pub.publish(msg)

    def _stop(self):
        self.cmd_pub.publish(Twist())
        self.shaper.reset()   # [P2] next motion ramps from true zero

    def _draw(self, frame, text, color, y=30):
        cv2.putText(frame,text,(10,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

    def _pub_pose(self, pd):
        if not pd.get('position_3d'): return
        pose=PoseStamped()
        pose.header.stamp=self.get_clock().now().to_msg()
        pose.header.frame_id='camera_link'
        x,y,z=pd['position_3d']
        pose.pose.position.x=x; pose.pose.position.y=y; pose.pose.position.z=z
        pose.pose.orientation.w=1.0
        self.pose_pub.publish(pose)
        # Also publish distance for the voice-system watchdog
        dist_msg = Float32()
        dist_msg.data = float(z)
        self.owner_dist_pub.publish(dist_msg)

    def _pub_frame(self, frame):
        try:
            ann=self.bridge.cv2_to_imgmsg(frame,'bgr8')
            ann.header.stamp=self.get_clock().now().to_msg()
            self.ann_pub.publish(ann)
        except: pass

    def destroy_node(self):
        self._stop(); super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    try:
        node=HumanTrackerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__=='__main__':
    main()
