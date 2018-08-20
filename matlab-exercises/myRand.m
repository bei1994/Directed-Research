function [s, a] = my(low, high)
    a = low + rand(3,4)*(high - low);
    s = sumOver(a);

function rel = sumOver(M)
    v = M(:);
    rel = sum(v);