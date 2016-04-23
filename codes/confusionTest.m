% compute for ten times for 1nn
clear
clc
confusionMatrixTemp = zeros(10, 10);
%k = 5;
%for i = 1 : 2
    [confusionMatrix, allDeterminedValue] = confusion_matrix_svm();
    %confusionMatrixTemp = confusionMatrixTemp + confusionMatrix;
    %confusionMatrix = confusionMatrix + confusion_matrix_svm();
%end
confusionMatrix
%confusionMatrix = confusionMatrixTemp / 2;
