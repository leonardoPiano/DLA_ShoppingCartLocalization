[train_img,test_img,train_labels,test_labels]=loadDataset();
%Load network
net = alexnet;
%analyzeNetwork(net);
inputSize = net.Layers(1).InputSize; %Take input size 
test_img=augmentedImageDatastore(inputSize,test_img); %Resize test img
inputSize = net.Layers(1).InputSize;
layers = net.Layers(1:end-3); %take last layers
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

pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),train_img, ...
    'DataAugmentation',imageAugmenter);

options = trainingOptions('sgdm', ...
    'MaxEpochs',25, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',test_img, ...
    'ValidationFrequency',3, ...
    'ExecutionEnvironment','gpu', ...
    'Verbose',false, ...
    'Plots','training-progress');
netTransferAlexNetTuned = trainNetwork(augimdsTrain,layers,options);
save('netTransferAlexNetTuned');






    


