% This function will retrive the data with specfic label. e.g. DataRetriver(1)
function[data filename] = DataRetriever(label) 
allData = csvread('/Users/guichengwu/Desktop/ecs 271 assignment 1/pendigits-train.csv');
dataSize = size(allData, 1);
count = 1;
data = zeros(dataSize, 17);
for i = 1:dataSize
    if (allData(i, 17) == label)
        data(count, :) = allData(i, :);
        count = count + 1;
    end
end
count = count - 1;
data = data(1:count,:);

filename = strcat('/Users/guichengwu/Desktop/ecs 271 assignment 1/train',num2str(label));
filename = strcat(filename, '.csv');

% write csv file 
csvwrite(filename, data);
end