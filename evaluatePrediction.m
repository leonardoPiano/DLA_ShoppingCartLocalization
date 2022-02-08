[train_img,test_img,train_labels,test_labels]=loadDataset();
YPred=load('ResNetPredSVM.mat').YPred;
[confMat,order] = confusionmat(test_labels,YPred);
recall=[];
precision=[];
for i =1:size(confMat,1)
    recall(i)=confMat(i,i)/sum(confMat(i,:));
    precision(i)=confMat(i,i)/sum(confMat(:,i));
end
accuracy = mean(YPred == test_labels);
recall(isnan(recall))=[];
precision(isnan(precision))=[];
Recall=sum(recall)/size(confMat,1);
Precision=sum(precision)/size(confMat,1);
Fscore=(2*Recall)*Precision/(Precision+Recall);
figure
confusionchart(test_labels,YPred);
display(accuracy,'Accuracy')
display(mean(Fscore),'Fscore');
display(mean(recall),'Recall')
display(mean(precision),'Precision');