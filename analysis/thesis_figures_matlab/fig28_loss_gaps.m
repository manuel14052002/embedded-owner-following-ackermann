%FIG28_LOSS_GAPS  Figure 28: distribution of durations without an owner fix.
c = thesis_cfg();
T = rdsheet('F28_LossGaps', 'A:B');
g = T.gap_s;
edges = logspace(log10(0.1), log10(120), 22);
fig = figure('Position', [100 100 640 280]);
histogram(g, edges, 'FaceColor', c.blue, 'EdgeColor', 'white', 'FaceAlpha', 1);
set(gca, 'XScale', 'log', 'Box', 'off'); grid on; hold on;
med = median(g);
gs = sort(g); pos = 1 + (numel(gs) - 1) * 0.90;          % Excel PERCENTILE (inclusive) definition
p90 = gs(floor(pos)) + (pos - floor(pos)) * (gs(min(ceil(pos), end)) - gs(floor(pos)));
yl = ylim;
xline(med, '--', sprintf('median %.1f s', med), 'Color', c.ink2, 'LabelOrientation', 'horizontal');
xline(p90, '--', sprintf('p90 %.0f s', p90), 'Color', c.ink2, 'LabelOrientation', 'horizontal');
ylim(yl);
xlabel('duration without owner fix (s, log scale)'); ylabel('episodes');
title(sprintf('Owner-loss episodes in completed follow runs (n = %d)', numel(g)));
fprintf('n = %d, median %.2f s, p90 %.2f s, max %.1f s; <1 s: %d, 1-5 s: %d, >=5 s: %d\n', numel(g), med, p90, max(g), ...
        sum(g < 1), sum(g >= 1 & g < 5), sum(g >= 5));
save_fig(fig, 'fig28_loss_gaps');
