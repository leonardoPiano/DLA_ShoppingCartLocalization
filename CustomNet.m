train_table=readtable('training_list.csv'); %Read train table
test_table=readtable("validation_list.csv"); %Read val table

train_files=train_table{:,'Var1'}; %Take only the image files
train_labels=train_table{:,'Var6'}; %Classification Labels
train_img=imageDatastore("images\"); %Store dataset on ImageDataStore
train_img.Files=train_files;
train_img.Labels=categorical(train_labels);

test_files=test_table{:,'Var1'}; %Take only the image files
test_labels=test_table{:,'Var6'}; %Classification Labels 
test_img=imageDatastore("images\");
test_img.Files=test_files;
test_img.Labels=categorical(test_labels);
inputSize=[128,128,3];
%train_img=augmentedImageDatastore(inputSize,train_img); %Resize dataset
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize,train_img, ...
    'DataAugmentation',imageAugmenter);
test_img=augmentedImageDatastore(inputSize,test_img); 
layers = [
    imageInputLayer(inputSize)
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(16)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',100, ...
    'Shuffle','every-epoch', ...
    'ValidationData',test_img, ...
    'ValidationFrequency',30, ...
    'ExecutionEnvironment','gpu', ...
    'Verbose',false, ...
    'Plots','training-progress');
net = trainNetwork(augimdsTrain,layers,options);
save('net');