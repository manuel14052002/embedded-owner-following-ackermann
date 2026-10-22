%FIG34_JETSON_LOAD  Figure 34: Jetson CPU, GPU and input power at the 15 W setting.
c = thesis_cfg();
T = rdsheet('F34_JetsonLoad', 'A:F');
s = T.sample / 1000;
fig = figure('Position', [100 100 700 340]);
subplot(2, 1, 1);
plot(s, movmean(T.cpu_mean_pct, 30), 'Color', c.blue, 'LineWidth', 1.3); hold on;
plot(s, movmean(T.gpu_pct, 30), 'Color', c.orange, 'LineWidth', 1.3);
ylabel('utilisation (%)'); ylim([0 105]); grid on; box off;
legend({'CPU (mean of cores)', 'GPU'}, 'Location', 'east', 'Box', 'off');
title('Jetson load at 15 W power mode (30-sample moving average)');
subplot(2, 1, 2);
plot(s, movmean(T.vdd_in_mw, 30) / 1000, 'Color', c.ink2, 'LineWidth', 1.3);
ylabel('input power (W)'); xlabel('tegrastats sample (thousands)'); grid on; box off;
fprintf('CPU mean %.1f%%, GPU mean %.1f%%, GPU idle %.0f%% of samples, power %.2f W, RAM %.0f MB, Tj max %.1f C\n', ...
        mean(T.cpu_mean_pct), mean(T.gpu_pct), 100*mean(T.gpu_pct == 0), mean(T.vdd_in_mw)/1000, mean(T.ram_mb), max(T.tj_c));
save_fig(fig, 'fig34_jetson_load');
