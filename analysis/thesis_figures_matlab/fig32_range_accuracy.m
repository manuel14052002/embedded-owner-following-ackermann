%FIG32_RANGE_ACCURACY  Figure 32: owner range error of camera and lidar vs tape ground truth.
c = thesis_cfg();
T = rdsheet('F32_RangeAccuracy', 'A:K');
tape = T{:, 'tape_m'};
ec = 100 * (T{:, 'cam_med_m'} - tape);            % camera error (cm)
el = 100 * (T{:, 'lid_rad_m (radial)'} - tape);    % lidar radial error (cm)
fig = figure('Position', [100 100 640 290]); hold on;
errorbar(tape - 0.06, ec, 100*T{:, 'cam_sd_m'}, 'o', 'Color', c.blue, 'MarkerFaceColor', c.blue, 'CapSize', 5, 'LineWidth', 1.2);
errorbar(tape + 0.06, el, 100*T{:, 'lid_sd_m'}, 'o', 'Color', c.orange, 'MarkerFaceColor', c.orange, 'CapSize', 5, 'LineWidth', 1.2);
yline(0, 'Color', c.ink2);
xlabel('tape-measured owner distance (m)'); ylabel('range error (cm)'); xticks(1:5); grid on; box off;
legend({sprintf('RGB-D camera (MAE %.1f cm, bias %+.1f cm)', mean(abs(ec)), mean(ec)), ...
        sprintf('2D lidar (MAE %.1f cm, bias %+.1f cm)', mean(abs(el)), mean(el))}, 'Location', 'northwest', 'Box', 'off');
title('Owner range error vs ground truth (S1 sweep r03, \pm1 SD)');
pc = polyfit(tape, T{:, 'cam_med_m'}, 1); pl = polyfit(tape, T{:, 'lid_med_m (forward)'}, 1);
fprintf('camera: MAE %.1f cm, RMSE %.1f cm, fit %.4f*tape %+.4f m\n', mean(abs(ec)), rms(ec), pc(1), pc(2));
fprintf('lidar : MAE %.1f cm, RMSE %.1f cm, bias %+.1f cm, fit %.4f*tape %+.4f m\n', mean(abs(el)), rms(el), mean(el), pl(1), pl(2));
save_fig(fig, 'fig32_range_accuracy');
