%MAKE_ALL_FIGURES  Regenerate every data figure of the TDK thesis from
%A2R-bot_TDK_Thesis_Figure_Data.xlsx. Run from this folder:  >> make_all_figures
scripts = {'fig05_ackermann', 'fig16_identification', 'fig17_gated_fixed', 'fig23_monitor_zones', ...
           'fig24_follow_sim', 'fig25_delay_loop', 'fig26_outcomes', 'fig27_owner_source', ...
           'fig28_loss_gaps', 'fig29_timeline', 'fig32_range_accuracy', 'fig33_latency', ...
           'fig34_jetson_load', 'circuit_calcs'};
failed = {};
for i = 1:numel(scripts)
    fprintf('\n=== %s ===\n', scripts{i});
    try
        run(scripts{i});
    catch err
        fprintf(2, 'FAILED %s: %s\n', scripts{i}, err.message);
        failed{end + 1} = scripts{i}; %#ok<SAGROW>
    end
    close all;
end
fprintf('\n%d of %d scripts succeeded.\n', numel(scripts) - numel(failed), numel(scripts));
if ~isempty(failed), fprintf('failed: %s\n', strjoin(failed, ', ')); end
