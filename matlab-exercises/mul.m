[row, col] = size(A);
P = zeros(row, col);
for i = 1:row*col
    P(i) = A(i) * A(i);
end

