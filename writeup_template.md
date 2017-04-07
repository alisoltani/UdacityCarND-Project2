## Traffic Sign Recognition
--

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/alisoltani/UdacityCarND-Project2/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook. 

* The size of training set is : Number of training examples = 34799
* The size of test set is : Number of testing examples = 12630
* The shape of a traffic sign image is : Image data shape = (34799, 32, 32, 3)
* The number of unique classes/labels in the data set is : Number of classes = 43


####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third, fourth and fifth code cell of the IPython notebook.  

In the third cell I am plotting a random example from each class to get a feeling of how each class looks like/feels. Each image has the label number and description associated with that number from signnames.csv as a title.

In the fourth cell, I wanted to see how the mean of all images in each class looks like. I decided on looking at the grayscaled version of each image, and plotting the mean. This would help give me a rough feeling how well each sign is aligned.

In cell five, I explore the number of images per class. This is important to look at, since not all classes have the same number of images and my training set may be become too biased towards those with more images. There are many classes with little number of images in them.


###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the sixth code cell of the IPython notebook. The first step is to shuffle the data so we don't always start with the same data and the parameters end up biased to the starting point. After that I make the images to grayscale. I first experimented with using color images in the network. There are a few cases where color may be important, like when comparing the sign for speed limit of 80 (which is has a red circle) and end of speed limit 80 (which is white and black), but overall in the tests that I did converting to grayscale gave performance improvements both in terms of speed and accuracy (around 2-3% better with grayscale). This is due to most of the information in the signs being in the shapes of those signs, and the actual amount of information in the colors was not that great.
I think that the colors in traffic signs are actually meant to grab our attention (signs with red in them seem to be very important for example) and should not be weighted equally when measuring accuracy. This was out of the current scope of the work, but I think it would be better to have penalty functions for missing more important signs (stop signs for example, children crossing, etc) where in those cases color conveys more relevant information.

Finally I normalized the images by making them between 0 and 1. This was done to as I was going to use zero mean and small variance for the weights, so I thouhgt it was necessary so that each random innovation had an actual affect on the output.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The data set provided was already split up into train.p, valid.p and test.p, and I used those data sets as my train, validation and test data sets. There were 34799 images in the training set, 4410 in the validation set, and 12630 in the testing set. 

In the seventh cell, I augment the data set in the cases where there are fewer than 400 images in that class. I chose this number after testing a few different because it seemed to be improve accruacy by enough without taking too much time. I think a better approach in general would be to have the number of images in each class match the actual statistics and frequencies of those traffic signs seen in real-life. The classes which have fewer than 400 images in them are completed to 400 by taking random images from those classes, averaging that sign with the mean calculated before, and then rotating it by a random number of degrees between 0 and 30. I decided against flipping because I felt that keep right and keep left signs would end up having problems.

After augmenting the test data set, the new dataset size is 36889. 


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the eighth cell of the ipython notebook. 

My final model was based on Lenet from the course, with some changes and consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x8 	|
| RELU					|												|
| Avg pooling	      	| 2x2 stride,  outputs 14x14x8 					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Flatten				| Input = 5x5x16. Output = 400					|
| Dropout				| Keep prob = 0.5 for test, 1 for valid			|
| Fully connected		| Input = 400. Output = 150						|
| RELU					|												|
| Dropout				| Keep prob = 0.5 for test, 1 for valid			|
| Fully connected		| Input = 150. Output = 92						|
| RELU					|												|
| Dropout				| Keep prob = 0.5 for test, 1 for valid			|
| Fully connected		| Input = 92. Output = 43						|
|						|												|
 
There were some changes compared to Lenet. First I included dropouts, since I noticed that there was too much overfitting where validation data accuracy was above 95% but test data accuracy was only 90-91%. I changed the first layer to average pooling instead of max pooling, and also slightly increased the number of classes in the hidden layers to improve performance. 

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the tenth cell of the ipython notebook. 

To train the model, I used an Adam optimizer to minimize the cross entropy between the output of the net and the one-hot labels. The data was fed in batches of 100, this was chosen in a trial and error fashion and seemed to hit the sweet spot regarding performance. I increased the number of epochs to 50 from the 10 in the course, as the accuracy was still improving before that. 

Finally I also added a dynamic rate, such that for every 5 epochs the rate would become 10% slower. This was done because the closer we get to the final valley, the slower we should move around. I tested with reducing 25% as well, but it took too long and needed an increase in the number of epochs.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the tenth and eleventh cell of the Ipython notebook.

My final model results were:
* training set accuracy of 98.5%
* validation set accuracy of 94.9% 
* test set accuracy of 93.5%

In different tests the validation and test accuracies were between 93-95% and 92.8 and 94% respectivly. The results here are from the last test iteration that was uploaded to git, but previous versions had slightly higher accuracy.

I started by adding a convulutional layer to Lenet from the course, but I was not able to achieve any better results. The two convulational layers in Lenet provide some basic pattern recognition, the first one tried to identify basic shapes like lines and circles and the second tries combining them. This gives a good basis to start our neural net and add fully connected layers afterwards. The first max pool was changed to average pool because I thought it would better represent the convolution. 

The dropout was included becase the training set accuracy was high, while validation and test was low. I tested 0.75 and 0.5, and 0.5 had better accuracy results.
 
Finally I decreased the standard deviation to 0.05 due the the normalization I had done, so each innovation is smaller which could be imporant when we are closer to the final answer, but may also result getting stuck in a local minima.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I included 10 German traffic signs I found on the web, and they are shown in the ipynb file. It can be noticed that the resolution of these images is slightly worse than those in the database, making them harder to classify. 

The first image might be difficult to classify because due to the lower resolution, it could easily be interpreted as a stay right/left. This is a case where having color would have helped as the blue in those signs would have contrasted with the red from this sign. 
The second image might be difficult to classify because children may be mistaken with pedetrian sign.
The third image could be difficult as the speed limit value may be mistaken to be for example 80 instead of 60.
The forth image is important to be able to model correctly as any mistake there could be fatal.
For the fifth image, I included the keep right signal so I was able to check how well I can identify a sign with different color.
In the sixth I added an extra speed limit, just to test and make sure it is working properly.
The seventh image is rotated and with a warped perspective, to test if it can identify the bumpy sign.
The eight image is not centered, so it can test how dependent we are on the poistion in the frame of the image.
The ninth image has an odd shape, it can help verify it can be modeled correctly.
Finally the tenth image is an end of speed limit sign, and it can be important to test as it can be easliy mistaken with the regular speed limit as I am not modeling color.


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 15th cell of the Ipython notebook, after pre-processing those images, in the 14th cell.

The model was able to correctly guess 8 of the 10 traffic signs, which gives an accuracy of 70%. This compares worsly to the accuracy on the test set of and can be attributed to the lower resolution and tougher conditions of those images. I was not able to get the confusion matrix to work, so I did not include it here in this project.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook. One strange thing was that when using the tf.nn.topk function, the probabilities in some cases were negative. I do not know who to interpret those values, and seems to me to be a bug in the code (possibly showing log probabilities instead of normal probabilities).
    
For image # 0  the probabilities are: [ 0.31233418  0.29712591  0.17957735  0.11697464  0.03608645] 
The predicted classes are: [31 37 18 29 27]
Correct image class is 31  
For the first image, the model is not completly sure that this is a wild animal crossing sign (probability of 0.31) and it could have been go straight or left too (0.29), but it identifies correctly and the image does contain a wild animal crossing sign. 

For image # 1  the probabilities are: [  9.85614359e-01   1.02695674e-02   3.47353192e-03   3.06325790e-04
   2.96291866e-04] 
The predicted classes are: [28 30 23 29 20]
Correct image class is 28 
In this case, the net is sure that it is a Children crossing sign. This is a correct assumption.

For image # 2  the probabilities are: [ 0.33193883  0.1118614   0.10254004  0.08789943  0.06569019] 
The predicted classes are: [ 5  3  8 10  7]
Correct image class is 3 
In this image the net is deciding between speed limit 60 and 80, with 80 having higher probability. This is not correct, and it was in fact 60.

For image # 3  the probabilities are: [ 0.94053763  0.01564077  0.01202679  0.00931493  0.00750925] 
The predicted classes are: [14 38 12  1  2]
Correct image class is 14 
In this image, the net is fairly sure that it is a stop sign, and it is actually a stop sign.

For image # 4  the probabilities are: [  1.00000000e+00   6.45131196e-17   2.03265790e-20   1.44491554e-20
   5.33868520e-24] 
The predicted classes are: [38 34 36 25 12]
Correct image class is 38 
In this image, the net is completly sure it is a keep right sign, and it is a keep right sign.

For image # 5  the probabilities are: [  9.88067746e-01   1.17382053e-02   1.61507633e-04   1.19632887e-05
   8.95661196e-06] 
The predicted classes are: [ 1  2  0 38  4]
Correct image class is 1
In this image, the net is fairly sure that this is a 30 kph speed limit sign, and it is actually such a sign.

For image # 6  the probabilities are: [ 0.34016445  0.27455905  0.08902305  0.07912602  0.05654135] 
The predicted classes are: [30 23 28 29 11]
Correct image class is 22 
In this image, the net thinks it is a beware of ice road sign, and it is not that sure either, but this is a misclassification and it is actually a bumpy road sign. This may be due to the warped perspective of the sign.

For image # 7  the probabilities are: [ 0.27939001  0.25583929  0.07711475  0.07691791  0.03349042] 
The predicted classes are: [40 12 25  1 18]
Correct image class is 40
It is unsure between a mandatory roundabout sign and priority road, but with a slightly hire probability for mandatory roundabout, which is correct.

For image # 8  the probabilities are: [  9.99998450e-01   1.59418732e-06   1.97885068e-08   8.82244588e-09
   7.82793474e-09] 
The predicted classes are: [12 40  1 13 15]
Correct image class is 12 
The net is fairly certain that this is a priority road sign, and it is actually such a sign.

For image # 9  the probabilities are: [ 0.22751142  0.14604031  0.10345463  0.10064253  0.05660363] 
The predicted classes are: [ 1  2  5  8 40]
Correct image class is 6 
In this image, the net is unsure what type of sign it is, and choses a 30kmph sign, which is wrong as it is an end of 80 kph sign. In this case color may have helped, as there is no red in this sign.

There is also a bar diagram in the .ipynb file that shows the information above.

I tried to visualize the network states, but I was not able to get the outputFeatureMap to work, and when I tried using conv1 or finding the correct tensor, it came up empty. I was not able to test this feature.

