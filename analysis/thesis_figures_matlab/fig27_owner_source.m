%FIG27_OWNER_SOURCE  Figure 27: owner position source while moving (completed follow runs).
c = thesis_cfg();
T = rdsheet('F27_OwnerSource', 'A:D');
T = T(~strcmp(T.run, 'Pooled') & ~isnan(T.keeper_s), :);   % runs only (no totals or notes)
tot = T.keeper_s + T.camera_s + T.none_s;
P = 100 * [T.keeper_s, T.camera_s, T.none_s] ./ tot;
[~, order] = sort(P(:, 3));                 % least no-fix at the bottom
P = P(order, :); names = T.run(order);

fig = figure('Position', [100 100 700 420]);
b = barh(P, 0.62, 'stacked', 'EdgeColor', 'white');
b(1).FaceColor = c.blue; b(2).FaceColor = c.aqua; b(3).FaceColor = c.grey;
for i = 1:size(P, 1)
    text(101, i, sprintf('%.0f%%', P(i, 3)), 'FontSize', 8);
end
set(gca, 'YTick', 1:numel(names), 'YTickLabel', names, 'FontSize', 8, 'Box', 'off');
xlim([0 108]); xlabel('share of moving time (%)');
pooled_none = 100 * sum(T.none_s) / sum(tot);
keeper_share = 100 * sum(T.keeper_s) / (sum(T.keeper_s) + sum(T.camera_s));
title(sprintf('Owner position source while moving (pooled no-fix %.1f%%)', pooled_none));
legend({'LiDAR owner keeper', 'camera (keeper silent)', 'no owner fix'}, 'Location', 'southoutside', 'Orientation', 'horizontal', 'Box', 'off');
fprintf('pooled no-fix %.1f%%, keeper share of tracked time %.1f%%\n', pooled_none, keeper_share);
save_fig(fig, 'fig27_owner_source');
