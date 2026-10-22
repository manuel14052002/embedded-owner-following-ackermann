%FIG29_TIMELINE  Figure 29: event timeline of run OUT3 (owner source, planner, safety monitor).
c = thesis_cfg();
T = rdsheet('F29_Timeline', 'A:D');
lanes = {'safety monitor', 'planner', 'owner source'};
col = containers.Map( ...
    {'keeper', 'follow', 'camera', 'none', 'idle/lost', 'manoeuvre', 'SLOW', 'STOP', 'PASS'}, ...
    {c.blue, c.blue, c.aqua, c.grey, c.grey, c.orange, c.orange, c.red, [0.843 0.902 0.969]});
fig = figure('Position', [100 100 760 260]); hold on;
for i = 1:height(T)
    y = find(strcmp(lanes, T.lane{i})) - 1;
    st = T.state{i};
    if isKey(col, st), cc = col(st); else, cc = c.grey; end
    rectangle('Position', [T.start_min(i), y - 0.35, T.end_min(i) - T.start_min(i), 0.7], 'FaceColor', cc, 'EdgeColor', 'none');
end
set(gca, 'YTick', 0:2, 'YTickLabel', lanes, 'Box', 'off'); ylim([-0.6 2.6]); xlim([0 max(T.end_min)]);
xlabel('time since motion start (min)');
title(sprintf('Event timeline of run OUT3 r01 (%.1f min moving)', max(T.end_min)));
h = gobjects(5, 1); labs = {'LiDAR keeper / follow', 'camera', 'none / idle', 'manoeuvre / SLOW', 'STOP'};
cs = {c.blue, c.aqua, c.grey, c.orange, c.red};
for k = 1:5, h(k) = patch(NaN, NaN, cs{k}, 'EdgeColor', 'none'); end
legend(h, labs, 'Location', 'southoutside', 'Orientation', 'horizontal', 'Box', 'off');
save_fig(fig, 'fig29_timeline');
