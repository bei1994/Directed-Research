function custom_write_bin(d1, d2, d3, d4, filename)

fid = fopen(filename, 'w+');
if fid < 0
    error('error opening %s file.', filename);
end

n1 = length(d1(:));
n2 = length(d2(:));
n3 = length(d3(:));

dim1 = size(d1);
dim2 = size(d2);
dim3 = size(d3);
dim4 = size(d4);

ndim1 = length(dim1);
ndim2 = length(dim2);
ndim3 = length(dim3);
ndim4 = length(dim4);


fwrite(fid, [n1, n2, n3], 'int16');
fwrite(fid, [ndim1, ndim2, ndim3, ndim4], 'int16');
fwrite(fid, [dim1, dim2, dim3, dim4], 'int16');

fwrite(fid, d1, 'char');
fwrite(fid, d2, 'single');
fwrite(fid, d3, 'int32');
fwrite(fid, d4, 'single');

fclose(fid);