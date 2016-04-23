% train svm to get the model
%trainData = csvread('/Users/guichengwu/Desktop/ecs 271 assignment 1/pendigits-train.csv');
trainData = csvread('/Users/guichengwu/Desktop/ecs 271 assignment 1/0vs8Source.csv');
%trainX = trainData(1:754, 1:16);
%trainX = trainData(775:1536, 1:16);
trainX = trainData(:, 1:16);
%trainY = trainData(1:754, 17);
trainY = trainData(:, 17);
classA = 0;
C = 0.05;
% C = 0.05;
epsion = 5*(10^(-5));
[trainX, trainY, alpha, beta, beta0, H] = SVMDigitClassifier(trainX, trainY, classA, C, epsion);

% get train data size
trainDataSize = size(trainX, 1);

% test svm with test data
testData = csvread('/Users/guichengwu/Desktop/ecs 271 assignment 1/0vs8Source.csv');
%testX = testData(1:21, 1:16);
testX = trainX;
%testY = testData(1:21, 17);
testY = trainY;
testSize = size(testX, 1);
processedTestX = GaussianKernel(testX, 1, 0);


classA = 0;
%labels for testY
for i = 1 : testSize
    if (testY(i) == classA)
        testY(i) = 1;
    else
        testY(i) = -1;
    end
end

%label for testX
labels = zeros(testSize, 1);
for i = 1 : testSize
    %determinedValue = processedTestX(i, :) * beta + beta0;
    determinedValue = 0;
    for j = 1 : trainDataSize
        if (alpha(j) ~= 0) 
            determinedValue = determinedValue + alpha(j)*trainY(j)*GKernel(testX(i), trainX(j), 1);
        end 
    end
    determinedValue = determinedValue + beta0;
    if (determinedValue >= 0)
            labels(i) = 1;
        else
            labels(i) = -1;
    end
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
    
    
