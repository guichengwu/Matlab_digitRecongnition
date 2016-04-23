function processedData = GaussianKernel(data, sigma, mu)
%set default value for sigma
if ~exist('sigma', 'var')
    sigma = 1;
end
%set default value for mu
if ~exist('mu', 'var')
    mu = 0;
end
dataSize = size(data, 1);
constantPrefix = sigma/(sqrt(2*pi));
constantNumber = 2 * (sigma^2);
for i = 1:dataSize
    temp = (-1)*((data(i,:) .* data(i, :) - mu)/ constantNumber);
    data(i,:) = exp(temp) * constantPrefix;
end
processedData = data;
end