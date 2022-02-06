%Read labels%
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
%Load network
net = vgg16;
%analyzeNetwork(net);
inputSize = net.Layers(1).InputSize; %Take input size 
%train_img=augmentedImageDatastore(inputSize,train_img); %Resize dataset
test_img=augmentedImageDatastore([128 128 3],test_img); %Resize dataset
inputSize = net.Layers(1).InputSize;

layers = [];
for i=1:length(net.Layers)
     if i==1
        layers(1)=imageInputLayer([128 128 3]);
     end
     layers(i)=net.Layers(i);
end
layers = layers(1:end-3);

%Freeze Layers
for i = 1:numel(layers)
        if isprop(layers(i),'WeightLearnRateFactor')
            layers(i).WeightLearnRateFactor = 0;
        end
        if isprop(layers(i),'WeightL2Factor')
            layers(i).WeightL2Factor = 0;
        end
        if isprop(layers(i),'BiasLearnRateFactor')
            layers(i).BiasLearnRateFactor = 0;
        end
        if isprop(layers(i),'BiasL2Factor')
            layers(i).BiasL2Factor = 0;
        end
end
%%%%
numClasses=16;
layers = [
    layers
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

inputSize=[128,128,3];
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),train_img, ...
    'DataAugmentation',imageAugmenter);

options = trainingOptions('sgdm', ...
    'MiniBatchSize',64, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',test_img, ...
    'ValidationFrequency',3, ...
    'ExecutionEnvironment','gpu', ...
    'Verbose',false, ...
    'Plots','training-progress');
netTransferVGGTuned = trainNetwork(augimdsTrain,layers,options);
save('netTransferVGGTuned');








    


