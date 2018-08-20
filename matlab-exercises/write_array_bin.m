function write_array_bin(A, filename)

fid = fopen(filename, 'w+');
if fid < 0
    error('error opening the file.');
end

fwrite(fid, A, 'double');
fclose(fid);