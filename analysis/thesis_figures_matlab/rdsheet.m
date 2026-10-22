function T = rdsheet(sheet, cols, hdrRow, lastRow)
%RDSHEET  Read one data table from the figure workbook.
%   T = rdsheet('F28_LossGaps', 'A:B')             header in row 4 (default)
%   T = rdsheet('F16_Identification', 'A:H', 11, 15)  header row 11, data to row 15
if nargin < 3, hdrRow = 4; end
if nargin < 4, lastRow = 20000; end
c = thesis_cfg();
parts = split(cols, ':');
rng = sprintf('%s%d:%s%d', parts{1}, hdrRow, parts{2}, lastRow);
T = readtable(c.xlsx, 'Sheet', sheet, 'Range', rng, 'VariableNamingRule', 'preserve');
% drop the empty rows below the table (first column missing)
first = T{:, 1};
if iscell(first) || isstring(first)
    keep = ~(ismissing(string(first)) | strlength(string(first)) == 0);
else
    keep = ~isnan(first);
end
T = T(keep, :);
end
