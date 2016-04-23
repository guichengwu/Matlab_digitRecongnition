% train svm to get the model
trainData = csvread('/Users/guichengwu/Desktop/ecs 271 assignment 1/pendigits-train.csv');
%trainData = csvread('/Users/guichengwu/Desktop/ecs 271 assignment 1/train.csv');
%trainData = csvread('/Users/guichengwu/Desktop/ecs 271 assignment 1/knnTestData.csv');
%trainX = trainData(1:1000, 1:16);
trainX = trainData(:, 1:16);
%trainX = trainData(775:1536, 1:16);
%trainY = trainData(1:1000, 17);
trainY0 = trainData(:, 17);
trainY1 = trainData(:, 17);
trainY2 = trainData(:, 17);
trainY3 = trainData(:, 17);
trainY4 = trainData(:, 17);
trainY5 = trainData(:, 17);
trainY6 = trainData(:, 17);
trainY7 = trainData(:, 17);
trainY8 = trainData(:, 17);
trainY9 = trainData(:, 17);
class0 = 0;
class1 = 1;
class2 = 2;
class3 = 3;
class4 = 4;
class5 = 5;
class6 = 6;
class7 = 7;
class8 = 8;
class9 = 9;
%C = 10000;
C = 0.05;
%C = 0.02;
epsion = 5*(10^(-5));
%sigma = 100; 92%
%sigma = 80;  93%
%sigma = 60; 94%
%sigma = 72;  93.4%
%sigma = 52; 93.9%
%sigma = 58; 94.2%
sigma = 59;
d = 4;
%k1 = 0.2;
%k2 = -2;
k1 = 0.2;
k2 = -2;
kernelType = 'gaussian';
%kernelType = 'degreePolynomial';
%kernelType = 'NeuralNetwork';

%run the svm classifier 0 vs rest; 1 vs rest; ..... 9 vs rest.
[trainX, trainY0, alpha0, beta00] = SVMDigitClassifier_1(trainX, trainY0, class0, C, epsion, sigma, d, k1, k2, kernelType);
[trainX, trainY1, alpha1, beta01] = SVMDigitClassifier_1(trainX, trainY1, class1, C, epsion, sigma, d, k1, k2, kernelType);
[trainX, trainY2, alpha2, beta02] = SVMDigitClassifier_1(trainX, trainY2, class2, C, epsion, sigma, d, k1, k2, kernelType);
[trainX, trainY3, alpha3, beta03] = SVMDigitClassifier_1(trainX, trainY3, class3, C, epsion, sigma, d, k1, k2, kernelType);
[trainX, trainY4, alpha4, beta04] = SVMDigitClassifier_1(trainX, trainY4, class4, C, epsion, sigma, d, k1, k2, kernelType);
[trainX, trainY5, alpha5, beta05] = SVMDigitClassifier_1(trainX, trainY5, class5, C, epsion, sigma, d, k1, k2, kernelType);
[trainX, trainY6, alpha6, beta06] = SVMDigitClassifier_1(trainX, trainY6, class6, C, epsion, sigma, d, k1, k2, kernelType);
[trainX, trainY7, alpha7, beta07] = SVMDigitClassifier_1(trainX, trainY7, class7, C, epsion, sigma, d, k1, k2, kernelType);
[trainX, trainY8, alpha8, beta08] = SVMDigitClassifier_1(trainX, trainY8, class8, C, epsion, sigma, d, k1, k2, kernelType);
[trainX, trainY9, alpha9, beta09] = SVMDigitClassifier_1(trainX, trainY9, class9, C, epsion, sigma, d, k1, k2, kernelType);
% Predict Process
% get train data size
trainDataSize = size(trainX, 1);

%testX = trainX;
%testY = trainY;

%load('svm.mat');

testData = csvread('/Users/guichengwu/Desktop/ecs 271 assignment 1/pendigits-train.csv');
%testX = testData(1:50, 1:16);
%testY = testData(1:50, 17);

testX = testData(:, 1:16);
testY = testData(:, 17);
testSize = size(testX, 1);

%label for testX
labels = zeros(testSize, 1);

allDeterminedValue = zeros(testSize, 10);
for i = 1 : testSize
    determinedValue = zeros(10, 1);
    for j = 1 : trainDataSize
        switch kernelType
            case 'NeuralNetwork'
                if (alpha0(j) ~= 0)
                    determinedValue(1) = determinedValue(1) + alpha0(j)*trainY0(j)*NeuralNetworkKernel(testX(i, :), trainX(j, :), k1, k2);
                end
                if (alpha1(j) ~= 0)
                    determinedValue(2) = determinedValue(2) + alpha1(j)*trainY1(j)*NeuralNetworkKernel(testX(i, :), trainX(j, :), k1, k2);
                end
                if (alpha2(j) ~= 0)
                    determinedValue(3) = determinedValue(3) + alpha2(j)*trainY2(j)*NeuralNetworkKernel(testX(i, :), trainX(j, :), k1, k2);
                end
                if (alpha3(j) ~= 0)
                    determinedValue(4) = determinedValue(4) + alpha3(j)*trainY3(j)*NeuralNetworkKernel(testX(i, :), trainX(j, :), k1, k2);
                end
                if (alpha4(j) ~= 0)
                    determinedValue(5) = determinedValue(5) + alpha4(j)*trainY4(j)*NeuralNetworkKernel(testX(i, :), trainX(j, :), k1, k2);
                end
                if (alpha5(j) ~= 0)
                    determinedValue(6) = determinedValue(6) + alpha5(j)*trainY5(j)*NeuralNetworkKernel(testX(i, :), trainX(j, :), k1, k2);
                end
                if (alpha6(j) ~= 0)
                    determinedValue(7) = determinedValue(7) + alpha6(j)*trainY6(j)*NeuralNetworkKernel(testX(i, :), trainX(j, :), k1, k2);
                end
                if (alpha7(j) ~= 0)
                    determinedValue(8) = determinedValue(8) + alpha7(j)*trainY7(j)*NeuralNetworkKernel(testX(i, :), trainX(j, :), k1, k2);
                end
                if (alpha8(j) ~= 0)
                    determinedValue(9) = determinedValue(9) + alpha8(j)*trainY8(j)*NeuralNetworkKernel(testX(i, :), trainX(j, :), k1, k2);
                end
                if (alpha9(j) ~= 0)
                    determinedValue(10) = determinedValue(10) + alpha9(j)*trainY9(j)*NeuralNetworkKernel(testX(i, :), trainX(j, :), k1, k2);
                end
            case 'degreePolynomial'
                if (alpha0(j) ~= 0)
                    determinedValue(1) = determinedValue(1) + alpha0(j) * trainY0(j) * DegreePolyKernel(testX(i,:), trainX(j, :), d);
                end
                if (alpha1(j) ~= 0)
                    determinedValue(2) = determinedValue(2) + alpha1(j) * trainY1(j) * DegreePolyKernel(testX(i,:), trainX(j, :), d);
                end
                if (alpha2(j) ~= 0)
                    determinedValue(3) = determinedValue(3) + alpha2(j) * trainY2(j) * DegreePolyKernel(testX(i,:), trainX(j, :), d);
                end
                if (alpha3(j) ~= 0)
                    determinedValue(4) = determinedValue(4) + alpha3(j) * trainY3(j) * DegreePolyKernel(testX(i,:), trainX(j, :), d);
                end
                if (alpha4(j) ~= 0)
                    determinedValue(5) = determinedValue(5) + alpha4(j) * trainY4(j) * DegreePolyKernel(testX(i,:), trainX(j, :), d);
                end
                if (alpha5(j) ~= 0)
                    determinedValue(6) = determinedValue(6) + alpha5(j) * trainY5(j) * DegreePolyKernel(testX(i,:), trainX(j, :), d);
                end
                if (alpha6(j) ~= 0)
                    determinedValue(7) = determinedValue(7) + alpha6(j) * trainY6(j) * DegreePolyKernel(testX(i,:), trainX(j, :), d);
                end
                if (alpha7(j) ~= 0)
                    determinedValue(8) = determinedValue(8) + alpha7(j) * trainY7(j) * DegreePolyKernel(testX(i,:), trainX(j, :), d);
                end
                if (alpha8(j) ~= 0)
                    determinedValue(9) = determinedValue(9) + alpha8(j) * trainY8(j) * DegreePolyKernel(testX(i,:), trainX(j, :), d);
                end
                if (alpha9(j) ~= 0)
                    determinedValue(10) = determinedValue(10) + alpha9(j) * trainY9(j) * DegreePolyKernel(testX(i,:), trainX(j, :), d);
                end
            otherwise
                if (alpha0(j) ~= 0)
                    determinedValue(1) = determinedValue(1) + alpha0(j)*trainY0(j)*GKernel(testX(i, :), trainX(j, :), sigma);
                end
                if (alpha1(j) ~= 0)
                    determinedValue(2) = determinedValue(2) + alpha1(j)*trainY1(j)*GKernel(testX(i, :), trainX(j, :), sigma);
                end
                if (alpha2(j) ~= 0)
                    determinedValue(3) = determinedValue(3) + alpha2(j)*trainY2(j)*GKernel(testX(i, :), trainX(j, :), sigma);
                end
                if (alpha3(j) ~= 0)
                    determinedValue(4) = determinedValue(4) + alpha3(j)*trainY3(j)*GKernel(testX(i, :), trainX(j, :), sigma);
                end
                if (alpha4(j) ~= 0)
                    determinedValue(5) = determinedValue(5) + alpha4(j)*trainY4(j)*GKernel(testX(i, :), trainX(j, :), sigma);
                end
                if (alpha5(j) ~= 0)
                    determinedValue(6) = determinedValue(6) + alpha5(j)*trainY5(j)*GKernel(testX(i, :), trainX(j, :), sigma);
                end
                if (alpha6(j) ~= 0)
                    determinedValue(7) = determinedValue(7) + alpha6(j)*trainY6(j)*GKernel(testX(i, :), trainX(j, :), sigma);
                end
                if (alpha7(j) ~= 0)
                    determinedValue(8) = determinedValue(8) + alpha7(j)*trainY7(j)*GKernel(testX(i, :), trainX(j, :), sigma);
                end
                if (alpha8(j) ~= 0)
                    determinedValue(9) = determinedValue(9) + alpha8(j)*trainY8(j)*GKernel(testX(i, :), trainX(j, :), sigma);
                end
                if (alpha9(j) ~= 0)
                    determinedValue(10) = determinedValue(10) + alpha9(j)*trainY9(j)*GKernel(testX(i, :), trainX(j, :), sigma);
                end
        end
    end
determinedValue(1) = determinedValue(1) + beta00;
determinedValue(2) = determinedValue(2) + beta01;
determinedValue(3) = determinedValue(3) + beta02;
determinedValue(4) = determinedValue(4) + beta03;
determinedValue(5) = determinedValue(5) + beta04;
determinedValue(6) = determinedValue(6) + beta05;
determinedValue(7) = determinedValue(7) + beta06;
determinedValue(8) = determinedValue(8) + beta07;
determinedValue(9) = determinedValue(9) + beta08;
determinedValue(10) = determinedValue(10) + beta09;

[M, I] = max(determinedValue);
labels(i) = I - 1;

allDeterminedValue(i, :) = determinedValue;
end

%calculate correct rate
correctCount = 0;
for i = 1: testSize
    if (testY(i) == labels(i))
        correctCount = correctCount + 1;
    end
end
correctRate = correctCount / testSize;
correctRate

% % confusion matrix
% confusionMatrix = computeMatrix(labels, testY);
