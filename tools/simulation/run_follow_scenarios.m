% RUN_FOLLOW_SCENARIOS  Simulated following behaviour of the deployed control law.
% Produces fig_sim_range_vs_speed.png, fig_sim_Lturn.png and prints steady-state ranges.
% SIMULATION ONLY (kinematic bicycle model, see a2rbot_follow_sim.m).
clear; close all;
speeds = [0.15 0.20 0.24 0.30];
figure('Position', [100 100 520 360]); hold on;
for k = 1:numel(speeds)
  [lg, P] = a2rbot_follow_sim(speeds(k), 'straight', 40, 0.1, 0, 0);
  plot(lg(:,1), lg(:,9), 'LineWidth', 1.2);
  late = lg(lg(:,1) > 25, 9);
  fprintf('owner %.2f m/s: range for t > 25 s  mean %.3f m  (min %.3f, max %.3f)\n', ...
          speeds(k), mean(late), min(late), max(late));
end
plot([0 40], [P.follow P.follow], 'k--');
xlabel('Time (s)'); ylabel('Robot-owner range (m)'); grid on; box on;
legend(arrayfun(@(s) sprintf('owner %.2f m/s', s), speeds, 'UniformOutput', false), 'Location', 'northwest');
title('Simulated following range (deployed pi\_goal law)');
fprintf('minimum turning radius L/tan(delta_max) = %.3f m\n', P.L / tan(P.dmax));
print('-dpng', '-r300', 'fig_sim_range_vs_speed.png');

figure('Position', [100 100 620 300]);
for k = 1:2
  Tc = [0.1 0.27]; Tc = Tc(k);
  lg = a2rbot_follow_sim(0.20, 'L', 45, Tc, 0.03, 1);
  subplot(1, 2, k); plot(lg(:,7), lg(:,8), 'k--', lg(:,2), lg(:,3), 'b', 'LineWidth', 1.2);
  axis equal; grid on; xlabel('x (m)'); ylabel('y (m)');
  title(sprintf('L-turn, update %.2f s', Tc)); legend('owner', 'robot', 'Location', 'northwest');
  fprintf('L-turn, Tc = %.2f s: max range %.3f m, final %.3f m, max steering %.1f deg\n', ...
          Tc, max(lg(:,9)), lg(end,9), rad2deg(max(abs(lg(:,6)))));
end
print('-dpng', '-r300', 'fig_sim_Lturn.png');
