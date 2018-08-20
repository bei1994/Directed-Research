function [o1, o2, o3, o4] = custom_read_bin(filename)

fid = fopen(filename, 'r+');
if fid < 0
    error('error opening %s file.', filename);
end

n = fread(fid, 3, 'int16');
ndim = fread(fid, 4, 'int16');
% fprintf(': %d', ndim);
dim1 = fread(fid, ndim(1), 'int16');
dim2 = fread(fid, ndim(2), 'int16');
dim3 = fread(fid, ndim(3), 'int16');
dim4 = fread(fid, ndim(4), 'int16');

o1 = fread(fid, n(1), '*char')';
o2 = fread(fid, n(2), 'single');
o3 = fread(fid, n(3), 'int32');
o4 = fread(fid, 'single');

o1 = reshape(o1, dim1');
o2 = reshape(o2, dim2');
o3 = reshape(o3, dim3');
o4 = reshape(o4, dim4');

fclose(fid);

