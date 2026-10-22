#!/usr/bin/env bash
# audit_a2rbot.sh — READ-ONLY. Changes nothing. Output: ~/a2rbot_audit.txt
PKG=~/ros2_ws/src/human_tracker_ros2/human_tracker_ros2
OUT=~/a2rbot_audit.txt
{
echo "=== A2RBOT audit $(date) ==="
echo; echo "--- 1. start script ---";            cat ~/start_a2rbot.sh 2>&1
echo; echo "--- 2. package files ---";          ls -la "$PKG"
echo; echo "--- 3. is multicue_owner.py original? ---"
grep -c "<<PATCH>>\|\[A2RBOT-LEARN\]" "$PKG/multicue_owner.py" 2>&1 | sed 's/^/patch markers: /'
md5sum "$PKG"/multicue_owner.py* 2>&1
echo; echo "--- 4. git state ---"
git -C ~/ros2_ws/src/human_tracker_ros2 status --short 2>&1 | head -30
git -C ~/ros2_ws/src/human_tracker_ros2 log --oneline -3 2>&1
echo; echo "--- 5. symlink install? ---"
ls -l ~/ros2_ws/install/human_tracker_ros2/lib/python3*/site-packages/human_tracker_ros2/multicue_owner.py 2>&1
echo; echo "--- 6. how the matcher is imported ---"
grep -rn "multicue_owner\|MultiCueMatcher(" "$PKG" --include=*.py | grep -v "^$PKG/multicue_owner"
echo; echo "--- 7. call sites ---"
grep -rn "\.enroll(\|finalize_enrollment\|set_owned\|update_galleries\|matcher\.score(\|\.score(frame" \
     "$PKG" --include=*.py | grep -v "^$PKG/multicue_owner"
echo; echo "--- 8. enrollment state / timeouts / frame counts ---"
grep -rn "ENROLL\|enroll_frames\|_enroll\|timeout\|30\.0\|>= *20\b" "$PKG/tracker_node.py" "$PKG/person_tracker.py" 2>&1
echo; echo "--- 9. CSV logging of cues ---"
grep -rn "DictWriter\|csv\|s_body3d\|\[.cues.\]\|\"cues\"" "$PKG" --include=*.py | grep -v "^$PKG/multicue_owner" | head -30
echo; echo "--- 10. python env ---"
which python3; python3 -c "import numpy, cv2; print('numpy', numpy.__version__, 'cv2', cv2.__version__)"
} > "$OUT" 2>&1
echo "Audit written to $OUT  (nothing was changed)"
