%Read labels%
train_table=readtable('training_list.csv'); %Read train table
test_table=readtable("validation_list.csv"); %Read val table

train_files=train_table{:,'Var1'}; %Take only the image files
train_labels=train_table{:,'Var6'}; %Classification Labels
train_img=imageDatastore("images\"); %Store dataset on ImageDataStore
train_img.Files=train_files;
numTrainImages = numel(train_labels);


test_files=test_table{:,'Var1'}; %Take only the image files
test_labels=test_table{:,'Var6'}; %Classification Labels 
test_img=imageDatastore("images\");
test_img.Files=test_files;
%Load alexNet network
net = alexnet;
inputSize = net.Layers(1).InputSize;
layersTransfer = net.Layers(1:end-3);
labels_array=categorical(train_labels);
numClasses = numel(categories(labels_array));
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];
train_img = augmentedImageDatastore(inputSize,train_img);
test_img = augmentedImageDatastore(inputSize,test_img);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',test_img, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');
netTransfer = trainNetwork(train_img,layers,options);
[YPred,scores] = classify(netTransfer,test_img);
accuracy = mean(YPred == test_labels);
display(accuracy)



