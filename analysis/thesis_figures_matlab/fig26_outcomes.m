%FIG26_OUTCOMES  Figure 26: success rates by test type with Wilson 95% intervals.
c = thesis_cfg();
T = rdsheet('F26_Outcomes', 'A:H', 5);
T = T(~isnan(T{:, 'runs n'}), :);           % drop the note row below the table
k = T{:, 'successes k'}; n = T{:, 'runs n'}; z = 1.96;
p = k ./ n;
ctr = (p + z^2 ./ (2*n)) ./ (1 + z^2 ./ n);
hw  = z * sqrt(p .* (1 - p) ./ n + z^2 ./ (4*n.^2)) ./ (1 + z^2 ./ n);
lo = ctr - hw; hi = ctr + hw;

fig = figure('Position', [100 100 640 230]);
barh(1:numel(p), 100*p, 0.55, 'FaceColor', c.blue, 'EdgeColor', 'none'); hold on;
errorbar(100*p, 1:numel(p), 100*(p - lo), 100*(hi - p), 'horizontal', 'k', 'LineStyle', 'none', 'CapSize', 8);
for i = 1:numel(p)
    text(103, i, sprintf('%d/%d', k(i), n(i)), 'FontSize', 9);
end
set(gca, 'YTick', 1:numel(p), 'YTickLabel', T{:, 'group'}, 'Box', 'off');
xlim([0 112]); xlabel('success rate (%) with Wilson 95% CI'); grid on;
title('Test outcomes, 20–22 Oct 2026 (24 runs)');
for i = 1:numel(p)
    fprintf('%-18s %d/%d = %.1f%%  [%.1f, %.1f]\n', T{i, 'group'}{1}, k(i), n(i), 100*p(i), 100*lo(i), 100*hi(i));
end
save_fig(fig, 'fig26_outcomes');
