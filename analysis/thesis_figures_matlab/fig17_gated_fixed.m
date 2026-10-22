%FIG17_GATED_FIXED  Figure 17: per-session owner/stranger separation, gated vs fixed fusion.
c = thesis_cfg();
T = rdsheet('F17_GatedFixed', 'A:F');
g = T.gated_sep; f = T.fixed_sep; n = T.n_rows; win = g > f;
fig = figure('Position', [100 100 470 400]); hold on;
plot([0 0.8], [0 0.8], '--', 'Color', c.ink2);
scatter(f(win), g(win), 12 + n(win)/150, c.blue, 'filled', 'MarkerFaceAlpha', 0.8, 'MarkerEdgeColor', 'white');
scatter(f(~win), g(~win), 12 + n(~win)/150, c.orange, 'filled', 'MarkerEdgeColor', 'white');
axis([0 0.8 0 0.8]); axis square; grid on; box off;
xlabel('owner/stranger separation, fixed-weight fusion'); ylabel('separation, gated fusion');
legend({'equal', sprintf('gated better (%d)', sum(win)), sprintf('fixed better (%d)', sum(~win))}, 'Location', 'northwest', 'Box', 'off');
title({'Gated vs fixed fusion, 38 sessions', '(marker area \propto frames)'});
d = g - f;
fprintf('gated better in %d/%d sessions; mean %.3f vs %.3f; mean diff %+.3f\n', sum(win), numel(g), mean(g), mean(f), mean(d));
try
    p = signrank(g, f);                      % Statistics and Machine Learning Toolbox
    fprintf('Wilcoxon signed-rank p = %.2g\n', p);
catch
    fprintf('signrank not available (Statistics Toolbox); sign test p = %.2g\n', 2*binocdf_local(min(sum(d>0), sum(d<0)), sum(d~=0)));
end
save_fig(fig, 'fig17_gated_fixed');

function p = binocdf_local(k, n)
p = sum(arrayfun(@(i) nchoosek(n, i), 0:k)) / 2^n;
end
