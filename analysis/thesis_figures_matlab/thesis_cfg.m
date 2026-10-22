function c = thesis_cfg()
%THESIS_CFG  Shared settings for the TDK thesis figure scripts.
%   Expects A2R-bot_TDK_Thesis_Figure_Data.xlsx in the parent folder of this script
%   (the Experiments folder). Figures are written to .\figures_matlab.
here = fileparts(mfilename('fullpath'));
c.xlsx = fullfile(here, '..', 'A2R-bot_TDK_Thesis_Figure_Data.xlsx');
if ~isfile(c.xlsx)
    error('thesis_cfg:noData', 'Workbook not found: %s', c.xlsx);
end
try   % R2025a+ may use the dark graphics theme; the thesis figures are light (print)
    s = settings;
    s.matlab.appearance.figure.GraphicsTheme.TemporaryValue = "light";
catch
end
c.out = fullfile(here, 'figures_matlab');
if ~isfolder(c.out), mkdir(c.out); end
c.blue   = [0.165 0.471 0.839];
c.orange = [0.922 0.408 0.204];
c.aqua   = [0.106 0.686 0.478];
c.red    = [0.890 0.286 0.282];
c.grey   = [0.725 0.722 0.698];
c.ink2   = [0.322 0.318 0.306];
end
