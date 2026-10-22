%FIG16_IDENTIFICATION  Figure 16 (cue reliability, acceptance by range) and Table 5 check.
c = thesis_cfg();
R = rdsheet('F16_Identification', 'A:E', 4, 9);      % 5 cues x 4 range bands
Acc = rdsheet('F16_Identification', 'A:H', 11, 15);  % acceptance by range
H = rdsheet('F16_Identification', 'A:F', 20, 25);    % Table 5 rules
bands = {'0–1.5', '1.5–2.5', '2.5–3.5', '3.5–6'};
Z = R{:, 2:5};
fig = figure('Position', [100 100 760 300]);
subplot(1, 2, 1);
imagesc(Z, [0 1]);
cm = interp1([0 1], [1 1 1; c.blue * 0.75], linspace(0, 1, 256)); colormap(gca, cm);
for i = 1:size(Z, 1)
    for j = 1:size(Z, 2)
        text(j, i, sprintf('%.2f', Z(i, j)), 'HorizontalAlignment', 'center', 'FontSize', 8, 'Color', [1 1 1] * (Z(i, j) > 0.6));
    end
end
set(gca, 'XTick', 1:4, 'XTickLabel', bands, 'YTick', 1:5, 'YTickLabel', {'ReID', 'negative', 'continuity', 'Kalman', 'hint'});
xlabel('owner range band (m)'); title('Cue reliability (Youden''s J)');
subplot(1, 2, 2);
Y = 100 * [Acc.strict, Acc.gated_cal, Acc.cascade];
b = bar(Y, 'EdgeColor', 'white'); b(1).FaceColor = c.grey; b(2).FaceColor = c.blue; b(3).FaceColor = c.orange;
set(gca, 'XTickLabel', bands, 'Box', 'off'); ylim([0 125]); xlabel('owner range band (m)'); ylabel('owner acceptance (%)');
legend({'strict ReID', 'gated, calibrated', 'deployed cascade'}, 'Location', 'southoutside', 'Orientation', 'horizontal', 'Box', 'off', 'FontSize', 7);
title('Held-out owner acceptance'); grid on;
% Table 5 with Wilson intervals
z = 1.96; wil = @(k, n) deal((k/n + z^2/(2*n) - z*sqrt(k/n*(1-k/n)/n + z^2/(4*n^2))) / (1 + z^2/n), ...
                            (k/n + z^2/(2*n) + z*sqrt(k/n*(1-k/n)/n + z^2/(4*n^2))) / (1 + z^2/n));
for i = 1:height(H)
    k = round(H{i, 'owner accept'} * 1110); [lo, hi] = wil(k, 1110);
    fprintf('%-34s owner %.1f%% [%.1f, %.1f]  volunteer frames %d/295\n', H.rule{i}, 100*H{i, 'owner accept'}, 100*lo, 100*hi, H{i, 'volunteer frames accepted'});
end
save_fig(fig, 'fig16_identification');
