function [confusionMatrix, allDeterminedValue] = SVMConfusionMatrixHelper(trainX, trainY, testX, testY)
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

%C = 0.05;
C = 0.05;
epsion = 5*(10^(-5));
sigma = 29;
%sigma = 1000;
d = 0.1;
k1 = 0.00001;
k2 = 1;
kernelType = 'gaussian';
trainDataSize = size(trainX, 1);

%run the svm classifier 0 vs rest; 1 vs rest; ..... 9 vs rest.
[trainX, trainY0, alpha0, beta00] = SVMDigitClassifier_1(trainX, trainY, class0, C, epsion, sigma, d, k1, k2, kernelType);
[trainX, trainY1, alpha1, beta01] = SVMDigitClassifier_1(trainX, trainY, class1, C, epsion, sigma, d, k1, k2, kernelType);
[trainX, trainY2, alpha2, beta02] = SVMDigitClassifier_1(trainX, trainY, class2, C, epsion, sigma, d, k1, k2, kernelType);
[trainX, trainY3, alpha3, beta03] = SVMDigitClassifier_1(trainX, trainY, class3, C, epsion, sigma, d, k1, k2, kernelType);
[trainX, trainY4, alpha4, beta04] = SVMDigitClassifier_1(trainX, trainY, class4, C, epsion, sigma, d, k1, k2, kernelType);
[trainX, trainY5, alpha5, beta05] = SVMDigitClassifier_1(trainX, trainY, class5, C, epsion, sigma, d, k1, k2, kernelType);
[trainX, trainY6, alpha6, beta06] = SVMDigitClassifier_1(trainX, trainY, class6, C, epsion, sigma, d, k1, k2, kernelType);
[trainX, trainY7, alpha7, beta07] = SVMDigitClassifier_1(trainX, trainY, class7, C, epsion, sigma, d, k1, k2, kernelType);
[trainX, trainY8, alpha8, beta08] = SVMDigitClassifier_1(trainX, trainY, class8, C, epsion, sigma, d, k1, k2, kernelType);
[trainX, trainY9, alpha9, beta09] = SVMDigitClassifier_1(trainX, trainY, class9, C, epsion, sigma, d, k1, k2, kernelType);

testSize = size(testX, 1);

%label for testX
labels = zeros(testSize, 1);

allDeterminedValue = zeros(testSize, 1);
for i = 1 : testSize
    determinedValue = zeros(10, 1);
    for j = 1 : trainDataSize
        if (alpha0(j) ~= 0)
            determinedValue(1) = determinedValue(1) + alpha0(j)*trainY0(j)*GKernel(testX(i, :), trainX(j, :), sigma);
            determinedValue(2) = determinedValue(2) + alpha1(j)*trainY1(j)*GKernel(testX(i, :), trainX(j, :), sigma);
            determinedValue(3) = determinedValue(3) + alpha2(j)*trainY2(j)*GKernel(testX(i, :), trainX(j, :), sigma);
            determinedValue(4) = determinedValue(4) + alpha3(j)*trainY3(j)*GKernel(testX(i, :), trainX(j, :), sigma);
            determinedValue(5) = determinedValue(5) + alpha4(j)*trainY4(j)*GKernel(testX(i, :), trainX(j, :), sigma);
            determinedValue(6) = determinedValue(6) + alpha5(j)*trainY5(j)*GKernel(testX(i, :), trainX(j, :), sigma);
            determinedValue(7) = determinedValue(7) + alpha6(j)*trainY6(j)*GKernel(testX(i, :), trainX(j, :), sigma);
            determinedValue(8) = determinedValue(8) + alpha7(j)*trainY7(j)*GKernel(testX(i, :), trainX(j, :), sigma);
            determinedValue(9) = determinedValue(9) + alpha8(j)*trainY8(j)*GKernel(testX(i, :), trainX(j, :), sigma);
            determinedValue(10) = determinedValue(10) + alpha9(j)*trainY9(j)*GKernel(testX(i, :), trainX(j, :), sigma);
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
    
    [M I] = max(determinedValue);
    labels(i) = I - 1;
    
    allDeterminedValue(i) = max(determinedValue);
end

confusionMatrix = computeMatrix(labels, testY);
end

