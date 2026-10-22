%CIRCUIT_CALCS  Eqs. 3–4 / Figure 12: battery divider and I2C idle voltage with mixed pull-ups.
c = thesis_cfg();
P = readmatrix(c.xlsx, 'Sheet', 'Circuit_Calcs', 'Range', 'B4:B11');
R2 = P(1); R3 = P(2); Vbat = P(3); R12 = P(4); Rpca = P(5); V33 = P(6); V5 = P(7); Vmax = P(8);
ratio = (R2 + R3) / R3; VA0 = Vbat / ratio;
Vidle = (V5*R12 + V33*Rpca) / (R12 + Rpca);   % Thevenin of the two pull-ups
fprintf('battery divider: ratio %.2f, V_A0 at %.1f V = %.3f V\n', ratio, Vbat, VA0);
fprintf('I2C idle voltage with %.1f k to %.1f V and %.1f k to %.1f V = %.2f V (ESP32-S3 abs. max %.1f V) -> %s\n', ...
        R12, V33, Rpca, V5, Vidle, Vmax, string(ifelse(Vidle > Vmax, 'EXCEEDED', 'ok')));
% sensitivity: idle voltage vs the breakout pull-up value
Rp = linspace(2.2, 47, 200); Vi = (V5*R12 + V33*Rp) ./ (R12 + Rp);
fig = figure('Position', [100 100 560 260]);
plot(Rp, Vi, 'Color', c.blue, 'LineWidth', 1.6); hold on; yline(Vmax, '--', 'ESP32-S3 abs. max 3.6 V', 'Color', c.red);
plot(Rpca, Vidle, 'o', 'MarkerFaceColor', c.orange, 'MarkerEdgeColor', 'none');
xlabel('PCA9685 breakout pull-up to 5 V (k\Omega)'); ylabel('I^2C idle voltage (V)'); grid on; box off;
title('I^2C idle level with mixed 3.3 V / 5 V pull-ups (R12/R13 = 4.7 k\Omega)');
save_fig(fig, 'circuit_i2c_idle');

function out = ifelse(cond, a, b)
if cond, out = a; else, out = b; end
end
