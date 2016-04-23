function confusionMatrix = knnForDigit(trainX, trainY, testX, testY, k)
testNum = size(testX, 1);
trainNum = size(trainX, 1);

labelClasses = unique(trainY);
labelNum = size(labelClasses, 1);

Result = zeros(testNum, 1);
labelCount = zeros(labelNum, 1);
dist = zeros(trainNum,1);
for i = 1:testNum
    testSample = testX(i,:);
    for j =1:trainNum
        trainSample = trainX(j,:);
        %d = bsxfun(@minus, trainSample, testSample);
        d = testSample-trainSample;
        %calculate distance
        dist(j) = norm(d, 2);
    end
    
   [Dummy, sortedDist] = sort(dist);
    
    labelCount(:) = 0;
    for j = 1:k
        labelIndex = find(trainY(sortedDist(j)) == labelClasses);
        labelCount(labelIndex) =  labelCount(labelIndex) + 1;
    end
    
    % determine the class of test data
    [dummy, determinedIndex] = max(labelCount);
    Result(i) = labelClasses(determinedIndex);
end
% correctNums = length(find(Result==testY));
% correctRate = correctNums / testNum;
confusionMatrix = computeMatrix(Result, testY);