%load dataset
[train_img,test_img]=loadDataset();
%Load Resnet18 network
net = resnet18;
inputSize = net.Layers(1).InputSize; %Take input size 
train_img=augmentedImageDatastore(inputSize,train_img); %Resize dataset
test_img=augmentedImageDatastore(inputSize,test_img); %Resize dataset
%Extract train and test features
layer='pool5';
train_features=activations(net,train_img,layer,'OutputAs','rows');
test_features=activations(net,test_img,layer,'OutputAs','rows');
%Fit img classifier
classifier = fitcecoc(train_features,train_labels);
%Predict
YPred = predict(classifier,test_features);
accuracy = mean(YPred == test_labels);
saveLearnerForCoder(classifier,'ResNet_SVM');
display(accuracy)




    


