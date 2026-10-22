%FIG24_FOLLOW_SIM  Figure 24: simulated following (from the workbook) and, optionally,
%a fresh re-simulation with the original MATLAB model a2rbot_follow_sim.m.
%   SIMULATION ONLY (kinematic bicycle model, deployed pi_goal law, D = 1.2 m).
c = thesis_cfg();
S = rdsheet('F24_SimFollow', 'A:E');
L = rdsheet('F24_SimFollow', 'N:T');
speeds = {'0.15', '0.20', '0.24', '0.30'}; al = [0.35 0.55 0.8 1];
fig = figure('Position', [100 100 780 320]);
subplot(1, 2, 1); hold on;
for i = 1:4
    y = S{:, 1 + i};
    if i < 4, plot(S.t_s, y, 'Color', [c.blue al(i)], 'LineWidth', 1.6);
    else,     plot(S.t_s, y, 'Color', c.orange, 'LineWidth', 1.6); end
end
yline(1.2, '--', 'stand-off D = 1.2 m', 'Color', c.ink2, 'LabelHorizontalAlignment', 'right');
xlabel('time (s)'); ylabel('owner–robot range (m)'); ylim([0 5]); grid on; box off;
legend(strcat(speeds, ' m/s'), 'Location', 'northwest', 'Box', 'off'); title('Owner walking straight away');
subplot(1, 2, 2); hold on;
plot(L{:, 'robot_x Tc0.10'}, L{:, 'robot_y Tc0.10'}, 'Color', c.blue, 'LineWidth', 1.6);
plot(L{:, 'robot_x Tc0.27'}, L{:, 'robot_y Tc0.27'}, 'Color', c.orange, 'LineWidth', 1.6);
plot(L.owner_x, L.owner_y, ':', 'Color', c.grey, 'LineWidth', 2.5);
axis equal; grid on; box off; xlabel('x (m)'); ylabel('y (m)');
legend({'robot, Tc = 0.10 s', 'robot, Tc = 0.27 s (measured)', 'owner path'}, 'Location', 'northwest', 'Box', 'off');
title('L-shaped walk, 0.20 m/s');
for i = 1:4
    fprintf('owner %s m/s: mean range for t > 25 s = %.3f m (workbook data)\n', speeds{i}, mean(S{S.t_s > 25, 1 + i}));
end
save_fig(fig, 'fig24_follow_sim');

% Optional: re-simulate with the original MATLAB model (includes the slow trim integral)
if exist('a2rbot_follow_sim', 'file') == 2
    for u = [0.15 0.20 0.24 0.30]
        lg = a2rbot_follow_sim(u, 'straight', 40, 0.1, 0, 0);
        fprintf('a2rbot_follow_sim.m  owner %.2f m/s: mean range t > 25 s = %.3f m\n', u, mean(lg(lg(:, 1) > 25, 9)));
    end
end
