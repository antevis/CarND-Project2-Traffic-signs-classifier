# **Traffic Sign Recognition** 



---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/bar_chart.png "Visualization"
[original_img]: ./examples/original1.png "Original Image"
[processed_imgs]: ./examples/processed6.png "Processed"
[augmented_imgs]: ./examples/aug_proc6.png "Augmented and Pre-processed"
[img0]: ./examples/0.png "Traffic Sign 1"
[img1]: ./examples/1.png "Traffic Sign 2"
[img2]: ./examples/2.png "Traffic Sign 3"
[img3]: ./examples/3.png "Traffic Sign 4"
[img4]: ./examples/5.png "Traffic Sign 5"
[img5]: ./examples/9.png "Traffic Sign 6"
[img6]: ./examples/12.png "Traffic Sign 7"
[img7]: ./examples/13.png "Traffic Sign 8"
[img8]: ./examples/16.png "Traffic Sign 9"
[img9]: ./examples/17.png "Traffic Sign 10"
[img10]: ./examples/20.png "Traffic Sign 11"
[img11]: ./examples/22.png "Traffic Sign 12"
[img12]: ./examples/28.png "Traffic Sign 13"
[img13]: ./examples/35.png "Traffic Sign 14"
[img14]: ./examples/36.png "Traffic Sign 15"
[img15]: ./examples/38.png "Traffic Sign 16"
[img16]: ./examples/40.png "Traffic Sign 17"
[fm]: ./examples/fm2.png "Priority Road Feature Map" 


[augmented_imgs]: ./examples/placeholder.png "Traffic Sign 5"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

Here is a link to my [project code](https://github.com/antevis/CarND-Project2-Traffic-signs-classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Basic summary of the data set.

The code for this step is contained in the 3rd code cell of the IPython notebook.  

I used standard Python library, NumPy and Pandas to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 samples
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3), that is, 32x32 pixels with depth of 3 channels.
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

The code for this step is contained in the 5th code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of frequencies of classes within dataset

![alt text][image1]

### Model Architecture design and testing

#### 1. Pre-processing

Numerous techniques of image pre-processing had been tested, including histogram equalization on RGB, HSV, HLS (all combinations of 1, 2 and 3 channels), CLACHE, chahe-bw etc. Cell 6 displays some of tested image transformations. Deliberately omitted grayscacling as color information may be relevant for better classification, as will be shown later. One of transformations (aux.normalize) is my own implementation that I created before discovering the vast variety of available possibilities with OpenCV. Turn out it is almost identical to `cv2.equalizeHist` histogram equalization, though at some extreme cases of maximum contrast the latter performs better.
Numerous training tests proved histogram equalization through all three channels works best.

Original image:

![alt text][original_img]

Some pre-processed results:

![alt text][processed_imgs]


#### 2. Data preparation

The [provided data](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip) have already been splitted to training, validation and test sets with the ratios of 67/9/24%, which seems ok and probably no any redistribution required.

To improve accuracy, training and validation sets have been inflated 6 times by adding 5 images with random affine transformations and brightness variations per each sample. This preserves relative frequencies among classes. The existing relative frequencies may be relevant for better performance as it probably reflects the real life relative frequencies. At least, test set has the same frequency distribution as the training one. This would probably improve performance deriving the inclination for the network to pick the more frequent option in case of uncertainty. Augmentation code itself inspired by Vivek Yadav, though brightness is my own (keeps values within the 0-255 range)

My final training set had 208794 number of images. My validation set and test set had 26460 and 12630 number of images. Test set had been deliberately left intact as we are interested in testin accuracy on real images rather than artificially augmented.

The 7th code cell of the IPython notebook contains the code for augmenting the data set. It utilizes the `augment` method implemented in  `helper.py` module.

After augmenting, all samples had been pre-processed with the chosen technique, which is histogram equalization.

Here are some examples of augmented and pre-processed images:

![alt text][augmented_imgs]

#### 3. Final model architecture.

The code for my final model is located in the 11th cell of the ipython notebook. In fact, there is a dedicated module `models.py`, containing all models been tested. There are a lot of them, including fancy implementations with merging several convolutional layers into one flattened. It turns out going deeper than 3 layers of convolution makes no sense with such a tiny dimentions of the input images.
The final picked archtecture is:

CNN with 5 hidden layers - 3 convolutional followed by 2 fully connected layers of 2048 and 1024 nodes respectively. 5x5 kernel with stride on 1 applied in the first convolutional layer, 3x3 kernels with strides of 1 at 2nd and 3rd. Max-pooling applied to all 3 convolutional layers. Dropout applied to all 5 hidden layers.

Implementation allows to retrieve layers' weights for further L2-regularization and model name for saving purposes.

The model defined as traffic_net_v2_full_dropout in models.py

My final model consisted of the following layers:

| Layer                     |     Description	        					| 
|:-------------------------:|:---------------------------------------------:| 
| Input                     | 32x32x3 RGB image   							| 
| Convolution 5x5           | 1x1 stride, SAME padding, outputs 32x32x64 	|
| ELU (exponential ReLU)    |												|
| Max pooling               | 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3           | 1x1 stride, VALID padding, outputs 14x14x128 	|
| ELU (exponential ReLU)    |                                               |
| Max pooling               | 2x2 stride,  outputs 7x7x128                  |
| Convolution 3x3           | 1x1 stride, VALID padding, outputs 6x6x192 	|
| ELU (exponential ReLU)    |                                               |
| Max pooling               | 2x2 stride,  outputs 3x3x192                  |
| Fully connected           | 2048        									|
| Fully connected           | 1024        									|
| Output, Fully connected	| 43        									|


#### 4. Training

The code for setting up models hyperparamters, placeholders and operations is located in cell 12.

Here we set learning rate, batch size, dropout, L2-regularization etc.

When defining placeholders and we define a dictionary of names for futher extraction when evaluating softmax probabilities later.

Actual training performed at cell 14 of the ipython notebook.

The epochs count is actually a terminal parameter of last resort just to stop training in case network fails to achieve any of other early stopping criteria. That is, training continues until no improvement in validation accuracy detected for chosen number of epochs.

To train the model, I used a pipeline located in cell 14.

#### 5. Finding solution

Numerous architectures of variable depth and breadth have been tested, including fancy implementations with merging separate convolutional layers into one flattened.
As mentioned before, turns out going deeper than 3 layers of convolution makes no sense with such a tiny dimentions of the input images.

The code for calculating the accuracy of the model is located in the 15th cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.9941
* validation set accuracy of 0.9823 
* test set accuracy of 0.973

As mentioned above, a lot of architectures had been tried, including simple multi-layer perceptron, which showed a surprisingly high validation accuracy of 88.2%
In fact, as can be seen in module `models.py`, there are about 17 models in total, all of them tried at least once.

Layers were added, removed, expanded by breadth, merged into single flattened, kernel sizes varied, max pooling applied or not to all or some specific layers. Such a tiny input image size demands to be really careful with max pooling, as it cuts the dimentionality by a factor of at least 4 with each step and would probably not be effective after the 3d layer of convolution.

Some architectures even failed to train at all, showing accuracy of 0.054, which is still twice better than random choice.:)

After all experiments, model with 3 convolutional + 2 fully connected layers yielded the best accuracy.

I set learning rate to a conventional 0.001 and picked Adam optimizer to do the optimization.
I set batch size to small value of 64 to trade accuracy against speed. As I understand, small batch size means more batches being generated per epoch wich means more weight ajustments to be done. 


### Test a Model on New Images

#### 1. Obtaining new traffic signs.

Here are 17 real-life Lithuanian traffic signs installed in the city of Vilnius:

![alt text][img0] ![alt text][img1] ![alt text][img2] ![alt text][img3] 
![alt text][img4] ![alt text][img5] ![alt text][img6] 
![alt text][img7] ![alt text][img8] ![alt text][img9] 
![alt text][img10] ![alt text][img11] ![alt text][img12] 
![alt text][img13] ![alt text][img14] ![alt text][img15]
![alt text][img16] 

The 4th image might be difficult to classify due to the fact that digit '6' in German signs looks quite different. 

#### 2. Model's predictions on these new traffic signs.

The code for making predictions on my final model is located in the 27th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			              |     Prediction	        					                  | 
|:---------------------:|:---------------------------------------------:| 
| Priority road    		| Priority road   								| 
| No passing     		| No passing 									|
| Ahead only			| Ahead only									|
| Speed limit (20km/h)	| Speed limit (20km/h)							|
| Speed limit (60km/h)	| Speed limit (80km/h)      					|

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. In fact, there are actually 17 samples in new signs set, and overall test accuracy on new images is 94.1%, that is, 16/17 and it appears that 'Speed limit 60' is the only misclassified sign.

#### 3. Describing the model certainty

The code for making predictions on my final model is located in the 31st cell of the Ipython notebook.

Top 5 Softmax probabilities for above signs

Vehicles over 3,5 metric tons prohibited

| Probability         	 |     Prediction	        					                  | 
|:---------------------:|:---------------------------------------------:| 
| .0612         		| Vehicles over 3,5 metric tons prohibited   	| 
| .0232     			| Speed limit (30km/h) 							|
| .0231					| No vehicles									|
| .0229	      			| Priority road			 			        	|
| .0228				    | End of speed limit (80km/h)      				|


Speed limit (20km/h)

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .0610         		| Speed limit (20km/h)                      	| 
| .0231     			| End of speed limit (80km/h) 					|
| .0230					| General caution								|
| .0229	      			| Vehicles over 3,5 metric tons prohibited		|
| .0229				    | Speed limit (80km/h)          				|


Priority road

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .0606         		| Priority road			 			        	| 
| .0231     			| Stop              							|
| .0230					| No entry  									|
| .0229	      			| Speed limit (80km/h)			 			   	|
| .0229				    | Speed limit (100km/h)         				|


Children crossing

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .0599         		| Children crossing			 			       	| 
| .0269     			| Roadwork             							|
| .0234					| General caution  								|
| .0230	      			| Road narrows on the right			 		   	|
| .0229				    | No vehicles                   				|


Speed limit (60km/h)

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .0458         		| Speed limit (80km/h)			 			   	| 
| .0451     			| Speed limit (60km/h) 							|
| .0232					| Pedestrians    								|
| .0230	      			| Vehicles over 3,5 metric tons prohibited		|
| .0229				    | General caution  								|

It can be seen that for correctly classified samples the network's confidence never really exceeds 6,1% for any particular class, though the top probaility beats all others by the factor of 2 as second places never exceed 3%.

Situation for the misclassified sample is more interesting as first two probabilities are almost on par: 4.58% vs. 4.51%. That means that the network considers the sign to be a Speed limit of 60 km/h or 80 km/h with almost equal probability, all due to the '6' digit being drawn differently in German training images and Lithuanian test images.

### Visualizing the Neural Network's State with Test Images

Visualization of the network state implemented in cells 32, 48, 50.

Here is the result of visualization of first convolutional layer feature map for the sign 'Priority road':

![alt text][fm]

It can be seen that the network obviously captured shapes, colors and spacial location of the features. For example, feature maps 4, 14, 32, 40 and 62 are excited about red color, while 22 captures low value pixels. It reacts violently to both black and dark blue at the center. Activation on 'Priority road' exposes that feature maps are probably more sensitive to negative slope (\) lines rather than positive ones. Feature maps 9 and 21 seems to be responsible for blue color.


