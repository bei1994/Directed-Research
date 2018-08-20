function A = read_dim_array_bin(filename, datatype)

% read dim info to result.

fid = fopen(filename, 'r+');
if fid < 0
    error('error opening %s file.', filename);
end

ndims = fread(fid, 1, datatype);
dims = fread(fid, ndims, datatype);
A = fread(fid, datatype);
A = reshape(A, dims');
fclose(fid);