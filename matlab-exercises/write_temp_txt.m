function write_temp_txt(filename)

title_1 = 'Climate Data for Nashville, TN';
title_2 = '(Average highs (F), lows(F), and precip(in)';
label_1 = ' High ';
label_2 = ' Low ';
label_3 = '    Precip ';
month_1 = {'Jan', 'Feb', 'March', 'April', 'May', 'June'};
month_2 = {'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'};
data_1 = [ 
    46 28 3.98
    51 31 3.7 
    61 39 4.88
    70 47 3.94
    78 57 5.08
    85 65 4.09
    ];
data_2 = [
    89 69 3.78
    88 68 3.27
    82 61 3.58
    71 49 2.87
    59 40 4.45
    49 31 4.53
    ];

fid  = fopen(filename, 'w+t');
if fid < 0
    error('error opening file\n');
end

fprintf(fid, '%s\n', title_1);
fprintf(fid, '%s\n', title_2);
fprintf(fid, '\n');

fprintf(fid, '       %s%s%s\n', label_1, label_2, label_3);
for i = 1:length(month_1)
    fprintf(fid, '%5s: ', month_1{i});
    fprintf(fid, '%5.2f, %5.2f, %5.2f\n', data_1(i, :));
end
fprintf(fid, '\n');
for i = 1:length(month_2)
    fprintf(fid, '%5s: ', month_2{i});
    fprintf(fid, '%5.2f, %5.2f, %5.2f\n', data_2(i, :));
end
fclose(fid);