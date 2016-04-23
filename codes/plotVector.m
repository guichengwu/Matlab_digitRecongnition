clear

trainData = csvread('C:\Users\DELL2103\Desktop\Winter\Maching_Learning\Supervised_learning_project\0vs8TestNoLabels.csv');
trainX = trainData(102, 1:16);
 X = zeros(8,1);
 Y = zeros(8,1);
 for i = 1 :2:16
     X((i+ 1)/2) = trainX(i);
     
 end
 
 for i = 2:2:16
     Y(i/2) = trainX(i);  
 end
 
 plot(X,Y)