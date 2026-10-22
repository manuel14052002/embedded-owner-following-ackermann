%FIG25_DELAY_LOOP  Figure 25 / Eq. 15: linearised range loop with perception delay.
%   d'(t) = v_o - k (d(t - tau) - D); owner steps to v_o at t = 1 s. Recomputed here
%   by Euler integration and compared with the workbook values.
c = thesis_cfg();
P = readmatrix(c.xlsx, 'Sheet', 'F25_DelayLoop', 'Range', 'B4:B6');  k = P(1); D = P(2); vo = P(3);
W = rdsheet('F25_DelayLoop', 'A:E', 8);
taus = [0.10 0.27 1.00 2.20]; dt = 0.005; T = 25;
cols = {c.blue, c.aqua, c.orange, c.red};
fig = figure('Position', [100 100 640 260]); hold on;
for j = 1:numel(taus)
    nd = round(taus(j) / dt); d = D * ones(nd + 1, 1);
    for i = 0:round(T / dt) - 1
        v = vo * (i*dt > 1);
        d(end + 1, 1) = d(end) + dt * (v - k * (d(end - nd) - D)); %#ok<AGROW>
    end
    d = d(nd + 2:end); t = (0:numel(d) - 1)' * dt;
    plot(t, d, 'Color', cols{j}, 'LineWidth', 1.5, 'DisplayName', sprintf('\\tau = %.2f s (k\\tau = %.2f)', taus(j), k*taus(j)));
    dw = W{:, 1 + j}; tw = W.t_s;
    fprintf('tau = %.2f s: max |MATLAB - workbook| = %.2e m\n', taus(j), max(abs(interp1(t, d, tw) - dw)));
end
yline(D + vo/k, '--', sprintf('D + v_o/k = %.2f m', D + vo/k), 'Color', c.ink2, 'HandleVisibility', 'off');
xlabel('time (s)'); ylabel('range d (m)'); ylim([0.6 2.4]); grid on; box off;
legend('Location', 'northeast', 'NumColumns', 2, 'Box', 'off');
title(sprintf('Range loop with perception delay (k = %.1f 1/s, owner steps to %.1f m/s)', k, vo));
fprintf('steady range %.3f m; stability limit tau < pi/(2k) = %.3f s\n', D + vo/k, pi/(2*k));
save_fig(fig, 'fig25_delay_loop');
