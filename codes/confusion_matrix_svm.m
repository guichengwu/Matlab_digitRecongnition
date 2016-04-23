% confusion matrix for ten-fold validation
% divide train data into ten pieces, using 9 to trian and one for test
function  [confusionMatrix, allDeterminedValue] = confusion_matrix_svm()
trainData = csvread('/Users/guichengwu/Desktop/ecs 271 assignment 1/pendigits-train.csv');
[dataSize, col] = size(trainData);
testSize = round(dataSize / 10);
trainSize = dataSize - testSize;
testX = zeros(testSize, 16);
testY = zeros(testSize, 1);
trainX = zeros(trainSize, 16);
trainY = zeros(trainSize, 1);

randRow = randsample(dataSize, testSize);
for i = 1 : testSize
    testX(i,:) =  trainData(randRow(i),1:16);
    testY(i) =  trainData(randRow(i),17);
    trainData(randRow(i), 1) = inf;
end

count = 1;
for i = 1 : dataSize
    if (trainData(i,1) ~= inf)
        trainX(count, 1:16) = trainData(i, 1:16);
        trainY(count) = trainData(i, 17);
        count = count + 1;
    end
end

% labelIndex = zeros(10,  1);
% trainYSize = size(trainY, 1);
% labelIndexCount = 1;
% for i = 1 : (trainYSize-1)
%     if (trainY(i) ~= trainY(i+1))
%         labelIndex(labelIndexCount) = i;
%         labelIndexCount = labelIndexCount + 1;
%     end        
% end
% labelIndex(10) = trainYSize;

%confusionMatrix = knnForDigit(trainX, trainY, testX, testY, k);
[confusionMatrix, allDeterminedValue] = SVMConfusionMatrixHelper(trainX, trainY, testX, testY);
end

  

