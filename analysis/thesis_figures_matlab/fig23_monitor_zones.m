%FIG23_MONITOR_ZONES  Figure 23 / Eq. 14: collision-monitor STOP and SLOW distances vs speed.
c = thesis_cfg();
P = readmatrix(c.xlsx, 'Sheet', 'F23_MonitorZones', 'Range', 'B4:B10');   % a b m c e vcap floor
a = P(1); b = P(2); m = P(3); k = P(4); e = P(5); vcap = P(6); vfloor = P(7);
v = linspace(0, 0.30, 100);
dstop = a + b*v + m; dslow = dstop + k*v + e;
fig = figure('Position', [100 100 640 270]); hold on;
hSlow = area(v, dslow, 'FaceColor', [0.992 0.890 0.839], 'EdgeColor', 'none');
hStop = area(v, dstop, 'FaceColor', [0.965 0.788 0.788], 'EdgeColor', 'none');
plot(v, dslow, 'Color', c.orange, 'LineWidth', 1.5); plot(v, dstop, 'Color', c.red, 'LineWidth', 1.5);
xline(vcap, '--', sprintf('v cap %.2f m/s', vcap), 'Color', c.ink2);
xline(vfloor, ':', sprintf('ESC floor %.3f', vfloor), 'Color', c.grey);
s = a + b*vcap + m; l = s + k*vcap + e;
plot(vcap, s, 'o', 'MarkerFaceColor', c.red, 'MarkerEdgeColor', 'none'); text(vcap - 0.005, s + 0.06, sprintf('%.2f m', s), 'HorizontalAlignment', 'right');
plot(vcap, l, 'o', 'MarkerFaceColor', c.orange, 'MarkerEdgeColor', 'none'); text(vcap - 0.005, l + 0.06, sprintf('%.2f m', l), 'HorizontalAlignment', 'right');
xlabel('commanded speed v (m/s)'); ylabel('obstacle distance (m)'); xlim([0 0.3]); ylim([0 1.3]); box off;
legend([hStop hSlow], {'STOP zone', 'SLOW zone'}, 'Location', 'northwest', 'Box', 'off');
title('Collision-monitor STOP and SLOW distances vs speed');
fprintf('at %.2f m/s: d_stop = %.3f m, d_slow = %.3f m\n', vcap, s, l);
save_fig(fig, 'fig23_monitor_zones');
