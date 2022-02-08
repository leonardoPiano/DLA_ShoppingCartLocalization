%Read labels%
[train_img,test_img,train_labels,test_labels]=loadDataset();
%Load network
net = resnet18;
%analyzeNetwork(net);
inputSize = net.Layers(1).InputSize; %Take input size 
test_img=augmentedImageDatastore(inputSize,test_img); %Resize dataset
inputSize = net.Layers(1).InputSize;
lgraph = layerGraph(net);
% Replace the last layers
lgraph = replaceLayer(lgraph,'fc1000',...
  fullyConnectedLayer(16,'Name','fcNew'));
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',...
  classificationLayer('Name','ClassificationNew'));
% Re-connect all the layers in the original order 
% by using the support function createLgraphUsingConnections
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

numClasses=16;

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
save('netTransferResNetTuned');


function lgraph = createLgraphUsingConnections(layers,connections)

lgraph = layerGraph();
for i = 1:numel(layers)
    lgraph = addLayers(lgraph,layers(i));
end

for c = 1:size(connections,1)
    lgraph = connectLayers(lgraph,connections.Source{c},connections.Destination{c});
end

end





    


