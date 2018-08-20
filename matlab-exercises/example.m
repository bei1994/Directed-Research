function [table, summa] = multab(n, m)

%Multiple table and get the sum.
%return the N by M table and double sum.

if nargin < 1
    error('must input at least one input argument.');
end

if nargin < 2
    m = n;
elseif ~isscalar(m) || m < 1 || m ~= fix(m)
    error('m must be posituve integer.');
end

if ~isscalar(n) || n < 1 || n ~= fix(n)
    error('n must be posituve integer.');
end

table = (1:n)' * (1:m);

if nargout == 2
    summa = sum(table(:));
end
