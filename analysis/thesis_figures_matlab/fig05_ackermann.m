%FIG05_ACKERMANN  Figure 5 / Eqs. 1–2: Ackermann geometry and minimum turning radius.
c = thesis_cfg();
P = readmatrix(c.xlsx, 'Sheet', 'F05_Geometry', 'Range', 'B4:B7');   % L, dmax(deg), track, r
L = P(1); dmax = deg2rad(P(2)); track = P(3);
R = L / tan(dmax);
fig = figure('Position', [100 100 560 360]); hold on; axis equal; axis off;
rectangle('Position', [-0.21 -0.155 0.425 0.31], 'FaceColor', [0.94 0.93 0.92], 'EdgeColor', c.ink2);
wheel = @(x, y, a) patch(x + [-0.06 0.06 0.06 -0.06]*cos(a) - [-0.02 -0.02 0.02 0.02]*sin(a), ...
                         y + [-0.06 0.06 0.06 -0.06]*sin(a) + [-0.02 -0.02 0.02 0.02]*cos(a), c.ink2, 'EdgeColor', 'none');
wheel(0, track/2, 0); wheel(0, -track/2, 0);
wheel(L, track/2, atan(L/(R - track/2))); wheel(L, -track/2, atan(L/(R + track/2)));
plot([0 0], [-0.2 R], ':', 'Color', c.grey); plot([L 0], [track/2 R], ':', 'Color', c.grey); plot([L 0], [-track/2 R], ':', 'Color', c.grey);
plot(0, R, 'o', 'MarkerFaceColor', c.blue, 'MarkerEdgeColor', 'none'); text(0.02, R + 0.03, 'ICR', 'Color', c.blue);
th = linspace(-pi/2, -pi/2 + 1.2, 50); plot(R*cos(th), R + R*sin(th), 'Color', c.blue, 'LineWidth', 1.6);
text(-0.05, R/2, sprintf('R_{min} = L / tan \\delta_{max} = %.3f m', R), 'HorizontalAlignment', 'right', 'Color', c.blue);
text(L/2, -0.05, sprintf('L = %.3f m', L), 'HorizontalAlignment', 'center');
text(L + 0.08, 0.16, sprintf('\\delta_{max} = %.0f°', P(2)));
xlim([-0.75 0.75]); ylim([-0.28 0.62]);
title('Kinematic bicycle / Ackermann geometry of the A2R-bot');
fprintf('R_min = %.3f m, U-turn width about %.2f m\n', R, 2*R + track);
save_fig(fig, 'fig05_ackermann');
