%Read labels%
train_table=readtable('training_list.csv'); %Read train table
test_table=readtable("validation_list.csv"); %Read val table

train_files=train_table{:,'Var1'}; %Take only the image files
train_labels=train_table{:,'Var6'}; %Classification Labels
train_img=imageDatastore("images\"); %Store dataset on ImageDataStore
train_img.Files=train_files;

test_files=test_table{:,'Var1'}; %Take only the image files
test_labels=test_table{:,'Var6'}; %Classification Labels 
test_img=imageDatastore("images\");
test_img.Files=test_files;
%Load Resnet18 network
net = resnet18;
inputSize = net.Layers(1).InputSize; %Take input size 
train_img=augmentedImageDatastore(inputSize,train_img); %Resize dataset
test_img=augmentedImageDatastore(inputSize,test_img); %Resize dataset
%Extract train and test features
train_features=activations(net,train_img,'pool5','OutputAs','rows');
test_features=activations(net,test_img,'pool5','OutputAs','rows');
%Fit img classifier
classifier = fitcecoc(train_features,train_labels);
%Predict
YPred = predict(classifier,test_features);
accuracy = mean(YPred == test_labels);
saveLearnerForCoder(classifier,'ResNet_SVM');
display(accuracy)




    


