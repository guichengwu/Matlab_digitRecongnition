%data = readtable('/Users/guichengwu/Desktop/ecs 271 assignment 1/pendigits-train.csv', 'Format', '%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%u', 'ReadVariableNames',false);
data = csvread('C:\Users\DELL2103\Desktop\Winter\Maching_Learning\Supervised_learning_project\pendigits-train.csv');
trainX = data(:,1:16);
trainY = data(:,17);

testData = csvread('C:\Users\DELL2103\Desktop\Winter\Maching_Learning\Supervised_learning_project\pendigits-train.csv');
%testData = readtable('/Users/guichengwu/Desktop/ecs 271 assignment 1/pendigits-test.csv',  'Format', '%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%u', 'ReadVariableNames',false);
testX = testData(:, 1:16);
testY = testData(:, 17);
k = 2;

knnForDigit(trainX, trainY, testX, testY, k);