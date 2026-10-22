function save_fig(fig, name)
%SAVE_FIG  Export a figure as a 200-dpi PNG into figures_matlab and close it.
c = thesis_cfg();
try, theme(fig, 'light'); catch, end   % light theme on R2025a+; ignored on older releases
f = fullfile(c.out, [name '.png']);
exportgraphics(fig, f, 'Resolution', 200, 'BackgroundColor', 'white');
fprintf('saved %s\n', f);
close(fig);
end
