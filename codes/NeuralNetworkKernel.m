% Neural Network kernel function
function [K] = NeuralNetworkKernel(xi, xj, k1, k2)
if ~exist('k1', 'var')
    k1 = 0.00001;
end
if ~exist('k2', 'var')
    k2 = 1;
end
K = tanh(k1*((xi/100)*(xj'/100)) + k2);
end