function write_dim_array_bin(A, filename)

% write dim info into the binary file.

fid = fopen(filename, 'w+');
if fid < 0
    error('error opening %s file.', filename);
end

dims = size(A);
ndims = length(dims);

fwrite(fid, ndims, 'double');
fwrite(fid, dims, 'double');
fwrite(fid, A, 'double');
fclose(fid);