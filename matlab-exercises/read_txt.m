function read_txt(filename)

fid = fopen(filename, 'r');
if fid < 0
    error('error opening the file\n');
end

tline = fgets(fid);
while ischar(tline)
    fprintf('%s', tline);
    tline = fgets(fid);
end
fprintf(fid, '%s', 'helloworld');
fclose(fid);