function confusionMatrix = computeMatrix(result, testY)
confusionMatrix = zeros(10, 10);
dataSize = size(testY,1);

for j = 1 : dataSize  
    confusionMatrix(testY(j)+1, result(j)+1) = confusionMatrix(testY(j)+1, result(j)+1) + 1;
end
confusionMatrix = confusionMatrix/dataSize;


