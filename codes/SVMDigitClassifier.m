function [trainX, trainY, alpha, beta, beta0, H] = SVMDigitClassifier(trainX, trainY, classA, C, epsion)
   dataSize = size(trainX, 1);
   trainColumn = size(trainX, 2);
   
% preprocess data label classA with "1", others label with "-1"
for i = 1:dataSize
    if(trainY(i) == classA) 
        trainY(i) = 1;
    else
        trainY(i) = -1;
    end
end

% preprocess the train data using GaussianKernel
%trainX = GaussianKernel(trainX);
trainX = GaussianKernel(trainX, 1, 0);

% initialize H matrix
H = zeros(dataSize, dataSize);
for i = 1:dataSize
    rowX1 = trainX(i, :);
    for j = 1:dataSize     
        %rowXT2 = trainX(j, :)';
        rowXT2 = trainX(j, :);
        %H(i, j) = trainY(i)*trainY(j)*rowX1*rowXT2;
        H(i, j) = trainY(i)*trainY(j)*GKernel(rowX1, rowXT2, 1);
    end
end

% parameters f, Aeq, beq, A, b, lb ub for quadratic solver
f = (-1)*ones(dataSize, 1);
Aeq = trainY';
% default C parameter is 0.05
if ~exist('C', 'var')
    C = 0.05;
end

if ~exist('epsion', 'var')
    epsion = 5*(10^(-5));
end
beq = 0;
A = diag(ones(dataSize, 1), 0);
b = C*ones(dataSize, 1);
lb = zeros(dataSize, 1);
ub = b;

% use quadratic programming function to get alpha
alpha = quadprog(H, f, A, b, Aeq, beq, lb, ub);
alpha0Count = 0;
%preprocess alpha, if alpha(i) is less than 1*e-6, then set alpha(i) = 0
for i = 1 : dataSize
    if (alpha(i) < epsion) 
        alpha(i) = 0;
        alpha0Count = alpha0Count + 1;
    end
end
% Use alpha to get beta
%initialize beta as zeros of length column of trainX
beta = zeros(trainColumn, 1);
for i = 1:dataSize
    if (alpha(i) ~= 0)
      beta = beta + alpha(i) * trainY(i) * trainX(i,:)';
    end
end

%calculate beta0
mu = C - alpha;
countBeta0 = 0;
beta0 = 0;
for i = 1:size(mu, 1)
    if (mu(i,1) ~= 0 && alpha(i) ~= 0) 
        beta0 = beta0 + (1 / trainY(i)) - (trainX(i, :) * beta);
        countBeta0 = countBeta0 + 1;
    end
end
beta0 = beta0 / countBeta0;
end
