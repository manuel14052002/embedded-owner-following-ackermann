function [log, P] = a2rbot_follow_sim(u, kind, T, Tc, sigma, seed)
%A2RBOT_FOLLOW_SIM  Kinematic (bicycle-model) simulation of the deployed FollowMe
%following law ("pi_goal", followme/direct_control.py) on the A2RBOT Ackermann base.
%
%   [log, P] = a2rbot_follow_sim(u, kind, T, Tc, sigma, seed)
%     u     owner walking speed [m/s]
%     kind  'straight' (owner walks away along x) or 'L' (4 m straight, then 90 deg left)
%     T     simulated time [s]
%     Tc    controller / perception update period [s] (0.1 nominal, 0.27 measured on 10 Oct)
%     sigma std. dev. of the owner position measurement noise [m]
%     seed  random seed
%   log columns: t, x, y, theta, v, delta, owner_x, owner_y, range
%
%   Parameters are those of config/followme_deploy.yaml (md5 112294b5...) and the
%   defaults in followme/config.py. The camera is assumed at the rear-axle origin.
%   This is a SIMULATION of the control law, not a measurement of the robot.

if nargin < 1, u = 0.2; end
if nargin < 2, kind = 'straight'; end
if nargin < 3, T = 40; end
if nargin < 4, Tc = 0.1; end
if nargin < 5, sigma = 0; end
if nargin < 6, seed = 0; end
try
  rng(seed);                 % MATLAB / recent Octave
catch
  randn('seed', seed);       % older Octave
end

P.L = 0.28;  P.dmax = deg2rad(30);        % wheelbase [m], max steering angle
P.follow = 1.2; P.k_lin = 0.8;            % stand-off [m], speed gain [1/s]
P.vmax = 0.24; P.vmin = 0.168; P.vturn = 0.199;   % [m/s]; vmin = ESC stall limit
P.turn_b = deg2rad(75); P.dead = deg2rad(3);
P.k_ang = 1.0; P.look = 0.9; P.tol = 0.10;
P.acc = 0.5; P.dec = 1.0; P.rate = deg2rad(90);   % [m/s^2], [m/s^2], servo slew [rad/s]
P.trim_ki = 0.4; P.trim_band = deg2rad(12); P.trim_max = deg2rad(6); P.trim_leak = 30;

dt = 0.01; N = round(T / dt);
x = 0; y = 0; th = 0; v = 0; delta = 0; trim = 0; tnext = 0;
log = zeros(N, 9);
for i = 1:N
  t = (i - 1) * dt;
  o = owner_path(t, u, kind);
  if t >= tnext - 1e-9
    dx = o(1) - x; dy = o(2) - y;
    rx =  cos(th) * dx + sin(th) * dy + sigma * randn;
    ry = -sin(th) * dx + cos(th) * dy + sigma * randn;
    d = hypot(rx, ry); b = atan2(ry, rx); dg = d - P.follow; h = Tc;
    if dg < P.tol
      vd = 0; dd = 0;
    elseif abs(b) > P.turn_b
      vd = P.vturn; dd = sign(b) * P.dmax;          % Ackermann cannot turn in place
    else
      vd = min(P.vmax, P.k_lin * dg) * max(0, cos(b));
      if vd < 1e-3, vd = 0; else, vd = max(vd, P.vmin); end
      if vd == 0 || abs(b) < P.dead
        dd = 0;
      else
        kappa = P.k_ang * 2 * sin(b) / max(dg, P.look);
        dd = clamp(atan(P.L * kappa), -P.dmax, P.dmax);
      end
    end
    trim = trim * max(0, 1 - h / P.trim_leak);       % slow servo-trim integral
    if v > 0 && abs(b) < P.trim_band && dg >= P.tol
      trim = clamp(trim + P.trim_ki * b * h, -P.trim_max, P.trim_max);
    end
    if vd > 0, dd = clamp(dd + trim, -P.dmax, P.dmax); end
    v = v + clamp(vd - v, -P.dec * h, P.acc * h);
    if vd > 0 && v > 0 && v < P.vmin, v = P.vmin; end
    step = P.rate * h; delta = clamp(dd, delta - step, delta + step);
    tnext = tnext + Tc;
  end
  x = x + v * cos(th) * dt; y = y + v * sin(th) * dt;
  th = th + v * tan(delta) / P.L * dt;
  log(i, :) = [t, x, y, th, v, delta, o(1), o(2), hypot(o(1) - x, o(2) - y)];
end
end

function o = owner_path(t, u, kind)
if strcmp(kind, 'straight')
  o = [2.0 + u * t, 0];
else
  s = u * t;
  if s < 4, o = [2.0 + s, 0]; else, o = [6.0, s - 4]; end
end
end

function y = clamp(x, lo, hi)
y = min(max(x, lo), hi);
end
