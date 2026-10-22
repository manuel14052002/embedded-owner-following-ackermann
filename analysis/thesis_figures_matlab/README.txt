A2R-bot TDK thesis - MATLAB figure scripts
===========================================

Thesis : Embedded_Owner_Following_TDK_Thesis_v4.docx
Data   : ..\A2R-bot_TDK_Thesis_Figure_Data.xlsx  (must stay in the parent folder)
Tested : MATLAB R2026b (no toolboxes required; signrank is used if the
         Statistics and Machine Learning Toolbox is present)

Run everything:   >> cd A2R-bot_TDK_MATLAB
                  >> make_all_figures
PNG files (200 dpi) are written to .\figures_matlab and the key numbers
are printed to the command window.

Script                   Thesis item              Workbook sheet
-----------------------  -----------------------  ---------------------
fig05_ackermann.m        Figure 5, Eqs. 1-2       F05_Geometry
fig16_identification.m   Figure 16, Table 5       F16_Identification
fig17_gated_fixed.m      Figure 17                F17_GatedFixed
fig23_monitor_zones.m    Figure 23, Eq. 14        F23_MonitorZones
fig24_follow_sim.m       Figure 24                F24_SimFollow (+ a2rbot_follow_sim.m)
fig25_delay_loop.m       Figure 25, Eq. 15        F25_DelayLoop
fig26_outcomes.m         Figure 26                F26_Outcomes
fig27_owner_source.m     Figure 27                F27_OwnerSource
fig28_loss_gaps.m        Figure 28                F28_LossGaps
fig29_timeline.m         Figure 29                F29_Timeline
fig32_range_accuracy.m   Figure 32                F32_RangeAccuracy
fig33_latency.m          Figure 33                F33_Latency
fig34_jetson_load.m      Figure 34                F34_JetsonLoad
circuit_calcs.m          Figure 12, Eqs. 3-4      Circuit_Calcs

Helpers: thesis_cfg.m (paths, colours, light theme), rdsheet.m (read a sheet
table), save_fig.m (PNG export).

Original simulation model (from A2RBOT_matlab_and_analysis.zip):
a2rbot_follow_sim.m, run_follow_scenarios.m. The thesis Figure 24 uses a Python
port of this model without the slow trim integral; both agree at 0.15 and
0.20 m/s (1.302 / 1.448 m). At 0.24 and 0.30 m/s the original MATLAB model gives
2.007 / 3.957 m versus 1.554 / 4.104 m for the port, because the owner then walks
at or above the 0.24 m/s speed cap and the range depends on the trim integral.

Diagram-only figures (1-4, 6-15, 18-22, 30-31) are drawings, schematics or
photos and have no numeric data.
