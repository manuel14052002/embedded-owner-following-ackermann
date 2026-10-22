%FIG33_LATENCY  Figure 33: perception latency per stage and CDF of total frame time.
c = thesis_cfg();
T = rdsheet('F33_Latency', 'A:G');
stages = {'t_segment', 'segmentation'; 't_reid', 're-identification'; 't_localize', 'localisation'; ...
          't_identify', 'identification'; 't_gesture', 'gesture'};
med = zeros(5, 1); p95 = zeros(5, 1);
for i = 1:5
    v = T{:, stages{i, 1}}; med(i) = median(v); p95(i) = prctile(v, 95);
end
fig = figure('Position', [100 100 760 290]);
subplot(1, 2, 1); hold on;
barh(5:-1:1, p95, 0.6, 'FaceColor', [0.843 0.902 0.969], 'EdgeColor', 'none');
barh(5:-1:1, med, 0.6, 'FaceColor', c.blue, 'EdgeColor', 'none');
set(gca, 'YTick', 1:5, 'YTickLabel', flipud(stages(:, 2)), 'Box', 'off'); xlabel('ms per frame');
legend({'p95', 'median'}, 'Location', 'southeast', 'Box', 'off'); title('Stage latency'); grid on;
subplot(1, 2, 2);
tot = sort(T.t_total); cdf = (1:numel(tot))' / numel(tot) * 100;
plot(tot, cdf, 'Color', c.blue, 'LineWidth', 2); hold on; xline(100, '--', 'Color', c.ink2);
text(103, 30, sprintf('%.0f%% of frames\n> 100 ms', 100*mean(tot > 100)), 'FontSize', 8, 'Color', c.ink2);
xlabel('total perception time (ms)'); ylabel('frames (%)'); title('Total latency CDF'); grid on; box off;
fprintf('total: median %.1f ms, p95 %.1f ms; segmentation share %.0f%%; median frame interval %.3f s\n', ...
        median(T.t_total), prctile(T.t_total, 95), 100*mean(T.t_segment)/mean(T.t_total), median(diff(T.t_mono)));
save_fig(fig, 'fig33_latency');
