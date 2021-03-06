
# **Traffic Sign Recognition**

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/train.jpg "train"
[image2]: ./examples/valid.jpg "valid"
[image3]: ./examples/test.jpg "test"
[image8]: ./images/GT1.png "Speed limit (60km/h)"
[image9]: ./images/GT2.png "Stop"
[image10]: ./images/GT3.png "Shared use path"
[image11]: ./images/GT4.png "Yield"
[image12]: ./images/GT5.png "Priority road"

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?  
34799
* The size of the validation set is ?  
4410
* The size of test set is ?  
12630
* The shape of a traffic sign image is ?  
(34799, 32, 32, 3)
* The number of unique classes/labels in the data set is ?  
43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing distributions of all the classes in train, validation and test data. The distributions in the three parts are similar.
![alt text][image1]
![alt text][image2]
![alt text][image3]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decide to convert the images to grayscale because the precondition of the grayscal is that the traffic sign is irrelevant to the color. More importantly, there is only one channel in the gray scale, which improves model to reduce the amount of computation.

Then I use ImageDataGenerator to generate additional data because over-fitting. More data should be improve the situation. As the same time, it costs much time.

As a last step, I normalize the image data because it can improve the speed of convergence and the numerical accuracy.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Gray image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|										|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU		            |            									|
| Max pooling			| 2x2 stride, outputs 5x5x16    				|
| Flatten				| outputs 400								    |
| Fully connected		| outputs 120								    |
| DROPOUT				|										|
| Fully connected		| outputs 84								    |
| DROPOUT				|										|
| Fully connected		| outputs 43								    |
| Softmax       		| outputs 43								    |



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I use an Adam optimizer with learning rate 0.001 which is appropriate value. Large learning rate will not find the best value while little learning brings slow training. The batch size is set to 128. Larger batch size costs much time and memory. The dropout is 0.5,which improves over-fitting. The number of epochs is 100. Actually, the model converges about after 89 epochs. Model need more epochs because of generated additional data.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?  
0.993
* validation set accuracy of ?  
0.970(EPOCH 89)
* test set accuracy of ?  
0.949

If a well known architecture was chosen:
* What architecture was chosen?  
I choose the LeNet to classify the images.

* Why did you believe it would be relevant to the traffic sign application?  
LeNet is simple and it works well on the Minist dataset and it is also a multi-classification task.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?    
The final model's accuracy the training, validation and test set are similar. It works well. If the training accuracy is much larger than the validation and test accuracy, it is over-fitting. If all the accuracy are small, it is under-fitting. we can add dropout and regularizer for the over-fitting while we can add more fully connected layers for the under-fitting.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image8]
<center>Speed limit (60km/h)</center>

![alt text][image9]
<center>Stop</center>

![alt text][image10]
<center>Shared use path</center>

![alt text][image11]
<center>Yield</center>

![alt text][image12]
<center>Priority road</center>

The first image might be difficult to classify because the quality of the image is very poor,and the third image might be difficult to classify because it has two signs.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 60km/h      		    | Go straight or right   	                    |
| Stop     			    | Bumpy road									|
| Shared use path		| Roundabout mandatory						    |
| Yield	      		    | Yield					 				        |
| Priority road			| Priority road      							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. The accuracy on the test is 94.9%. There is a great difference between them. There are two reasons. One is that the classes of images is too small. The other is that lenet do not work well on noises situations.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is not sure that it is a Go straight or right sign, but the image is the sign of 60km/h. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
|0.349787|Go straight or right|
|0.130837|Children crossing|
|0.0805637|Right-of-way at the next intersection|
|0.0743875|Traffic signals|
|0.0574738|Pedestrians|


For the second image, the model is almost certain that it is a Bumpy road sign. But it is really a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
|0.999991|Bumpy road|
|3.73003e-06|No passing|
|2.45016e-06|Priority road|
|1.63958e-06|Dangerous curve to the right|
|6.5758e-07|Keep left|

For the third image, the model is almost certain that it is a Roundabout mandatory sign. But it is the sign of Shared use path. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
|0.895995|Roundabout mandatory|
|0.0608556|Keep right|
|0.00851481|Go straight or right|
|0.00832206|Go straight or left|
|0.00630474|Road work|

For the forth image, the model is certain that it is Yield sign. It really is. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
|1.0|Yield|
|1.60298e-16|Speed limit (60km/h)|
|9.7522e-22|Keep right|
|6.69195e-22|Priority road|
|1.50874e-23|No passing|

For the fifth image, the model is almost certain that it is Priority road. It really is. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
|0.999966|Priority road|
|2.58625e-05|Roundabout mandatory|
|8.25617e-06|Stop|
|5.66665e-08|Keep right|
|5.30576e-08|No entry|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
The feature maps learn many high level features from the images. For instance, the shape of the sign, and the edges of numbers on the sign.


```python

```
