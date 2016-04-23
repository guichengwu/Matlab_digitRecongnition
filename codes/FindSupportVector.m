trainData = csvread('/Users/guichengwu/Desktop/ecs 271 assignment 1/pendigits-train.csv');
trainX = trainData(:, 1:16);
trainY = trainData(:, 17);
class0 = 9;

%C = 0.05;
C = 0.1;
epsion = 5*(10^(-5));
sigma = 100;
d = 0.1;
k1 = 0.00001;
k2 = 1;
kernelType = 'gaussian';

[trainX, trainY0, alpha, beta0] = SVMDigitClassifier_1(trainX, trainY, class0, C, epsion, sigma, d, k1, k2, kernelType);

trainDataSize = size(trainX, 1);

labels = zeros(trainDataSize, 1);


supportVectorIndex = 1;
determinedValue = zeros(trainDataSize, 1);

supportVectorRange = find(trainY == class0);
startIndex = supportVectorRange(1);
endIndex = startIndex + size(supportVectorRange, 1) - 1;
for i = startIndex : endIndex
    for j = 1 : trainDataSize
        if (alpha(j) ~= 0)
            switch kernelType
                case 'NeuralNetwork'                    
                    determinedValue(i) = determinedValue(i) + alpha(j) * trainY0(j) * NeuralNetworkKernel(trainX(i, :), trainX(j, :), k1, k2);
                case 'degreePolynomial'                    
                    determinedValue(i) = determinedValue(i) + alpha(j) * trainY0(j) * DegreePolyKernel(trainX(i,:), trainX(j, :), d);                    
                otherwise
                    determinedValue(i) = determinedValue(i) + alpha(j) * trainY0(j) * GKernel(trainX(i, :), trainX(j, :), sigma);
            end
        end
    end
    determinedValue(i) = determinedValue(i) + beta0;
    
    if (determinedValue(i) >= 0.9 && determinedValue(i) <= 1.1)
        supportVectorIndex = i;
        %determinedValue(i)
        disp('The support vector index is:');
        i
        
        break;
    end    
end
    drawData = trainX(supportVectorIndex, :);
    drawX = zeros(8, 1);
    drawY = zeros(8, 1);
    for i = 1:2:16
        drawX((i+1) / 2) = drawData(i);
    end
    for i = 2:2:16
        drawY(i / 2) = drawData(i);
    end
    
    % draw support vector
    plot(drawX, drawY);
