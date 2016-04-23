% Degree polynomic kernel function
function [K] = DegreePolyKernel(xi, xj, d)
if ~exist('d', 'var')
    d = 0.1;
end

K = (1 + (xi/100) *(xj'/100))^d;
end