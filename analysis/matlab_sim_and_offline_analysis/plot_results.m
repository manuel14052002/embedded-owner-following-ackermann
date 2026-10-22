% PLOT_RESULTS  Figures from the measured A2RBOT data (files in ./data, exported from the
% robot logs of 10 Oct 2026 and 17-23 Sep 2026). Every number plotted here is measured.
clear; close all;
D = 'data';

% 1. Calibrated cue reliability versus owner range
fid = fopen(fullfile(D, 'reliability_by_range.csv')); fgetl(fid);
c = textscan(fid, '%s %f %f %f', 'Delimiter', ',', 'EmptyValue', NaN); fclose(fid);
cues = unique(c{1}, 'stable'); mid = (c{2} + c{3}) / 2;
figure('Position', [100 100 520 340]); hold on; mk = 'osd^v';
for k = 1:numel(cues)
  m = strcmp(c{1}, cues{k});
  plot(mid(m), c{4}(m), ['-' mk(k)], 'LineWidth', 1.2);
end
xlabel('Owner range (m)'); ylabel('Calibrated reliability c_i'); ylim([0 1.05]); grid on; box on;
legend(cues, 'Location', 'southwest'); title('Cue reliability calibrated on block A');
print('-dpng', '-r300', 'fig_reliability_vs_range.png');

% 2. Held-out comparison of the five decision rules
fid = fopen(fullfile(D, 'heldout_results.csv')); fgetl(fid);
h = textscan(fid, '%q %f %f %f %f %f %f', 'Delimiter', ',', 'EmptyValue', NaN); fclose(fid);
figure('Position', [100 100 620 340]);
bar([h{4} h{6}] * 100); set(gca, 'XTickLabel', h{1}, 'XTickLabelRotation', 20);
ylabel('Frames (%)'); legend('owner accepted', 'volunteer accepted', 'Location', 'east');
grid on; title('Held-out test: owner 1110 frames, volunteer 295 frames');
print('-dpng', '-r300', 'fig_heldout_rules.png');

% 3. Owner acceptance by range on the held-out owner frames
A = dlmread(fullfile(D, 'acceptance_by_range.csv'), ',', 1, 0);
figure('Position', [100 100 520 340]);
bar((A(:,1) + A(:,2)) / 2, A(:,4:8) * 100);
xlabel('Owner range (m)'); ylabel('Owner accepted (%)'); grid on;
legend({'strict', 'fixed', 'gated', 'gated (calibrated)', 'cascade'}, 'Location', 'southwest');
title('Owner acceptance by range (held-out)');
print('-dpng', '-r300', 'fig_acceptance_by_range.png');

% 4. Per-frame latency and frame interval at 15 W
Tm = dlmread(fullfile(D, 'timing_per_frame.csv'), ',', 1, 0);
interval = diff(Tm(:,1)) * 1000;
figure('Position', [100 100 520 340]); hold on;
s1 = sort(Tm(:,7)); s2 = sort(interval);
plot(s1, (1:numel(s1)) / numel(s1), 'LineWidth', 1.2);
plot(s2, (1:numel(s2)) / numel(s2), 'LineWidth', 1.2);
xlabel('Time (ms)'); ylabel('Cumulative fraction'); grid on; box on; xlim([0 600]);
legend('perception per frame', 'interval between processed frames', 'Location', 'southeast');
fprintf('perception: median %.1f ms, p95 %.1f ms | frame interval: median %.1f ms, p95 %.1f ms\n', ...
        median(s1), s1(ceil(0.95 * numel(s1))), median(s2), s2(ceil(0.95 * numel(s2))));
print('-dpng', '-r300', 'fig_latency_cdf.png');

% 5. September system: per-session separation, gated vs fixed (labels = system verdicts)
S = dlmread(fullfile(D, 'sept_session_separation.csv'), ',', 1, 0);
figure('Position', [100 100 420 380]);
plot(S(:,4), S(:,3), 'ko', 'MarkerFaceColor', 'k'); hold on; plot([0 0.8], [0 0.8], 'k--');
axis([0 0.75 0 0.75]); axis square; grid on;
xlabel('Median separation, fixed-weight'); ylabel('Median separation, gated');
title(sprintf('Sep 2026 system: gated above diagonal in %d of %d sessions', sum(S(:,3) > S(:,4)), size(S,1)));
print('-dpng', '-r300', 'fig_sept_separation.png');
