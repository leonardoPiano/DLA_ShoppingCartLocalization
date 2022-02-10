## Problem Introduction
The location of the shopping cart, is a useful activity in order to understand the behaviors and interests
of customers within the stores. Such information can be used to improve the management of the store
and to provide personalized services to the customers. For instance, location information can be used to
infer where the customers spend more time, which areas of the store they prefer. The problem can be
addressed as an image-based location problem, developing an algorithm that given an image taken in a
known space, allows you to infer the position from which the image was taken.
## Dataset
The dataset is taken from the Machine Learning Challange (2018) of the University of Catania.
E. Spera, A. Furnari, S. Battiato, G. M. Farinella, 
Egocentric Shopping Cart Localization, International Conference on Pattern Recognition (ICPR), 2018
http://iplab.dmi.unict.it/EgocentricShoppingCartLocalization/
## Methodology
We test three different methodologies, training a network from  scratch,
Feature extraction and Fine Tuning.
We perform Feature Extraction employing ResNet-18,Vgg16 and AlexNet , 
the extracted feature are then fed to an SVM for the classification.
In Fine Tuning, we Tuned ResNet-18 and AlexNet.

##  Results
The following results are obtained on the validation dataset.
#### Feature Extraction

Network | Accuracy | Precision | Recall | F1-Score | 
--------|----------|-----------|--------|----------|
ResNet  | 0.916    | 0.921     | 0.906  | 0.913    |
AlexNet  | 0.919    | 0.929     | 0.908  | 0.918    |
AlexNet  | 0.919    | 0.929     | 0.908  | 0.918    |
VggNet  | 0.905    | 0.918     | 0.898  | 0.908    |


#### Fine Tuning (Sgd)
Network | Accuracy | Precision | Recall | F1-Score | 
--------|----------|-----------|--------|----------|
ResNet  | 0.921    | 0.917     | 0.882  | 0.899    |
AlexNet  | 0.852    | 0.854     | 0.843  | 0.849    |
#### Fine Tuning (Adam)
Network | Accuracy | Precision | Recall | F1-Score | 
--------|----------|-----------|--------|----------|
ResNet  | 0.955    | 0.954     | 0.950  | 0.952    |
AlexNet  | 0.829    | 0.832     | 0.839  | 0.835    |



#### Network From Scratch (Sgd)
Network | Accuracy | Precision | Recall | F1-Score | 
--------|----------|-----------|--------|----------|
Custom  | 0.758    | 0.770     | 0.709  | 0.738    |

#### Network From Scratch (Adam)
Network | Accuracy | Precision | Recall | F1-Score | 
--------|----------|-----------|--------|----------|
Custom  | 0.894    | 0.900     | 0.875  | 0.887    |







