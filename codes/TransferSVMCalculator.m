sourceData = csvread('/Users/guichengwu/Desktop/ecs 271 assignment 1/0vs8Source.csv');
targetData = csvread('/Users/guichengwu/Desktop/ecs 271 assignment 1/0vs8Target.csv');

sourceX = sourceData(:, 1:16);
sourceY = sourceData(:, 17);
targetX = targetData(:, 1:16);
targetY = targetData(:, 17);
class = 0;
epsion = 5e-5;
C = 0.05;
sigma = 80;
[sourceX, sourceY, sourceAlpha, sourceBeta, sourceBeta0] = SVMDigitClassifier(sourceX, sourceY, class, C, epsion);
[targetX, targetY, targetAlpha, targetBeta, targetBeta0] = SVMDigitClassifier(targetX, targetY, class, C, epsion);


% calculate new target beta0
sourceSize = size(sourceData, 1);


targetDataSize = size(targetData, 1);
targetBeta0Sum = 0;

for i = 1:targetDataSize
    if (targetAlpha(i) ~= 0)
      temp = targetY(i) * (targetX(i, :) * sourceBeta + sourceBeta0) + (targetY(i) * targetX(i, :) * targetBeta);
    end
    temp = (1-temp) / targetY(i);
    targetBeta0Sum = targetBeta0Sum + temp;
end

targetBeta0 = targetBeta0Sum / targetDataSize;

% test svm with test data
testData = csvread('/Users/guichengwu/Desktop/ecs 271 assignment 1/0vs8TestNoLabels.csv');
testSize = size(testData, 1);
processedTestX = GaussianKernel(testData, sigma, 0);
   
%label for testX
labels = zeros(testSize, 1);
derterminedValue = zeros(testSize, 1);
for i = 1 : testSize
         derterminedValue(i) = processedTestX(i, :) * targetBeta + targetBeta0;
    if (derterminedValue(i) >= 0)
        labels(i) = 1;
    else
        labels(i) = -1;
    end
end


% for i = 1:testSize
%     derterminedValue(i) = 0;
%     for j =1:targetDataSize
%         if (targetAlpha(j) ~= 0)
%           derterminedValue(i) = derterminedValue(i) + targetAlpha(j) * targetY(j) * GKernel(testData(i,:), targetX(j,:), 1);
%         end
%     end
%     derterminedValue(i) = derterminedValue(i) + targetBeta0;
%     if (derterminedValue(i) >= eps)
%         labels(i) = 1;
%     else
%         labels(i) = -1;
%     end
% end
correctCount = 0;
for i = 1:101
    if (labels(i) == -1)
        correctCount = correctCount + 1;
    end
end

for i = 102:202
    if (labels(i) == 1)
        correctCount = correctCount + 1;
    end
end

correctRate = correctCount / 202;
correctRate

result = zeros(testSize, 1);
for i = 1:testSize
    if (labels(i) == 1) 
        result(i) = 0;
    end
    if (labels(i) == -1)
        result(i) = 8;
    end
end
  
outputfile = '/Users/guichengwu/Desktop/ecs 271 assignment 1/0vs8TestResult.csv';
csvwrite(outputfile, result);
        