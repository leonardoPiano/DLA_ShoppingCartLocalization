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
net = resnet18;
%analyzeNetwork(net);
inputSize = net.Layers(1).InputSize; %Take input size 
%train_img=augmentedImageDatastore(inputSize,train_img); %Resize dataset
test_img=augmentedImageDatastore(inputSize,test_img); %Resize dataset
inputSize = net.Layers(1).InputSize;
%layers = net.Layers(1:end-3);

lgraph = layerGraph(net);
% 2. Replace the last few layers
lgraph = replaceLayer(lgraph,'fc1000',...
  fullyConnectedLayer(16,'Name','fcNew'));
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',...
  classificationLayer('Name','ClassificationNew'));
% 4. Re-connect all the layers in the original order 
%    by using the support function createLgraphUsingConnections
layers = lgraph.Layers;
connections = lgraph.Connections;
lgraph = createLgraphUsingConnections(layers,connections);
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
%{
layers = [
    layers
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];
%}

pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),train_img, ...
    'DataAugmentation',imageAugmenter);

options = trainingOptions('sgdm', ...
    'MaxEpochs',10, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',test_img, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');
netTransfer = trainNetwork(augimdsTrain,lgraph,options);
save('netTransfer');

load('netTransferResNetTuned20epoche.mat','netTransferUpdated','augimdsTrain','lgraph','test_img');
options = trainingOptions('sgdm', ...
    'MaxEpochs',5, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',test_img, ...
    'ValidationFrequency',3, ...
    'ExecutionEnvironment','gpu', ...
    'Verbose',false, ...
    'Plots','training-progress');
netTransferUpdated = trainNetwork(augimdsTrain,layerGraph(netTransferUpdated),options);
save('netTransferUpdated');






function lgraph = createLgraphUsingConnections(layers,connections)

lgraph = layerGraph();
for i = 1:numel(layers)
    lgraph = addLayers(lgraph,layers(i));
end

for c = 1:size(connections,1)
    lgraph = connectLayers(lgraph,connections.Source{c},connections.Destination{c});
end

end





    


