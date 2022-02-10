function [train_img,test_img,train_labels,test_labels]=loadDataset()

train_table=readtable('training_list.csv'); %Read train table
test_table=readtable("validation_list.csv"); %Read val table

train_files=train_table{:,'Var1'}; %Take only the image files
train_labels=categorical(train_table{:,'Var6'}); %Classification Labels
train_img=imageDatastore("images\"); %Store dataset on ImageDataStore
train_img.Files=train_files;
train_img.Labels=train_labels;

test_files=test_table{:,'Var1'}; %Take only the image files
test_labels=categorical(test_table{:,'Var6'}); %Classification Labels 
test_img=imageDatastore("images\");
test_img.Files=test_files;
test_img.Labels=test_labels;

end