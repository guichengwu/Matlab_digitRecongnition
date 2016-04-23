%Gaussian Kernel
function [K] = GKernel(xi, xj, sigma) 
    if ~exist('sigma', 'var')
        sigma = 100;
    end
    constantNumber = 2 * (sigma^2);
     K = 1 / exp(((xi - xj) * (xi - xj)') / constantNumber); 
end