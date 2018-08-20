function [A, count] = read_bin_file(filename, datatype)

fid = fopen(filename, 'r+');
if fid < 0
    error('error opening the file.');
end

[A, count] = fread(fid, datatype);
fclose(fid);