function [trainX, trainY, alpha, beta0] = SVMDigitClassifier_1(trainX, trainY, classA, C, epsion, sigma, d, k1, k2, kernelType)
if ~exist('kernelType', 'var')
    kernelType = 'gaussian';
end
if ~exist('sigma', 'var')
    sigma = 29;
end

if ~exist('d', 'var')
    d = 0.01;
end
if ~exist('k1', 'var')
    k1 = 0.00001;
end
if ~exist('k2', 'var')
    k2 = 1;
end
dataSize = size(trainX, 1);

% preprocess data label classA with "1", others label with "-1"
label1Size = 0;
for i = 1:dataSize
    if(trainY(i) == classA)
        trainY(i) = 1;
        label1Size = label1Size + 1;
    else
        trainY(i) = -1;
    end
end


% initialize H matrix
H = zeros(dataSize, dataSize);
for i = 1:dataSize
    rowX1 = trainX(i, :);
    for j = 1:dataSize
        %rowXT2 = trainX(j, :)';
        rowX2 = trainX(j, :);
        %H(i, j) = trainY(i)*trainY(j)*rowX1*rowXT2;
        %H(i, j) = trainY(i)*trainY(j)*GKernel(rowX1, rowX2, sigma);
        switch kernelType
            case 'NeuralNetwork'
                H(i, j) = trainY(i)*trainY(j)*NeuralNetworkKernel(rowX1, rowX2, k1, k2);
            case 'degreePolynomial'
                H(i, j) = trainY(i) * trainY(j) * DegreePolyKernel(rowX1, rowX2, d);
            otherwise
                H(i, j) = trainY(i)*trainY(j)*GKernel(rowX1, rowX2, sigma);
        end
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
%preprocess alpha, if alpha(i) is less than 5*e-5, then set alpha(i) = 0
for i = 1 : dataSize
    if (alpha(i) < epsion)
        alpha(i) = 0;
        alpha0Count = alpha0Count + 1;
    end
end

%calculate beta0, using yt * f(xt) = 1
sumBeta0 = 0;
labelN1Count = 0;
label1Count = 0;
for t = 1: dataSize
    tempSum = 0;
    yt = trainY(t);
    if (yt == -1)
        labelN1Count = labelN1Count + 1;
    end
    if (yt == 1)
        label1Count = label1Count + 1;
    end
    if ((labelN1Count >= label1Size) && (label1Count >label1Size))
        break;
    end
    if (((yt == -1) && labelN1Count < label1Size) || ((yt == 1) && label1Count < label1Size))
        for i = 1 : dataSize
            switch kernelType
                case 'NeuralNetwork'
                    temp = alpha(i) * trainY(i) * NeuralNetworkKernel(trainX(t,:), trainX(i, :), k1, k2);
                case 'degreePolynomial'
                    temp = alpha(i) * trainY(i) * DegreePolyKernel(trainX(t,:), trainX(i, :), d);
                otherwise
                    temp = alpha(i) * trainY(i) * GKernel(trainX(t,:), trainX(i, :), sigma);
            end
            tempSum = tempSum + temp;
        end
        beta0 = (1 / yt) - tempSum;
        sumBeta0 = sumBeta0 + beta0;
    end
end

%beta0 = sumBeta0 / dataSize;
beta0 = sumBeta0 / (2 * label1Size);
end