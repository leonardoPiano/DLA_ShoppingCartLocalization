%Load Data
[train_img,test_img,train_labels,test_labels]=loadDataset();
%#Load trained network 
net=load('net100epoche.mat');
net.test_files=test_img.Files;
inputSize=net.net.Layers(1).InputSize;
test_img=augmentedImageDatastore(inputSize,test_img);
YPred=classify(net.net,test_img);
save('CustomNetPred.mat',"YPred");
