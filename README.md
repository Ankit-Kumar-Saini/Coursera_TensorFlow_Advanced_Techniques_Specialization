# TensorFlow: Advanced Techniques Specialization
- Taught by: [Laurence Moroney](https://www.coursera.org/instructor/lmoroney)


### Table of Contents
1. [Instructions to use the repository](#Instruction)
2. [My Learnings from the Specialization](#Learning)
3. [Programming Assignments](#Programming)
4. [Results](#Result)


## Instructions to use the repository<a name="Instruction"></a>
- Clone this repository to use. It contains all my work for this specialization. All the code base, screenshots, and images are taken from unless specified, [TensorFlow: Advanced Techniques Specialization on Coursera](https://www.coursera.org/specializations/tensorflow-advanced-techniques). 
- `Note`: The solutions uploaded in this repository are only for reference when you got stuck somewhere. Please don't use these solutions to pass the programming assignments.


## My Learnings from the Specialization<a name="Learning"></a>
This specialization from coursera consists of four courses. Below are my learnings from individual courses.
- [Course1: Custom Models, Layers, and Loss Functions with TensorFlow](https://www.coursera.org/learn/custom-models-layers-loss-functions-with-tensorflow?specialization=tensorflow-advanced-techniques)
	- Build models that produce multiple outputs (including a Siamese network) using the Functional API
	- Build custom loss functions (including the contrastive loss function used in a Siamese network)
	- Build custom layers using existing standard layers, customized network layer with a lambda layer and explored activation functions for custom layers
	- Build custom classes instead of using the Functional or Sequential APIs
	- Build models that can be inherited from the TensorFlow Model class, and build a residual network (ResNet) through defining a custom model class

- [Course2: Custom and Distributed Training with TensorFlow](https://www.coursera.org/learn/custom-distributed-training-with-tensorflow?specialization=tensorflow-advanced-techniques)
	- Learned about the difference between the eager and graph modes in TensorFlow
	- Build custom training loops using GradientTape and TensorFlow Datasets to gain more flexibility and visibility with the model training
	- Got an overview of various distributed training strategies, and practice working with a strategy that trains on multiple GPU cores, and another that trains on multiple TPU cores

- [Course3: Advanced Computer Vision with TensorFlow](https://www.coursera.org/learn/advanced-computer-vision-with-tensorflow?specialization=tensorflow-advanced-techniques)
	- Build image classification, image segmentation, object localization, and object detection models
	- Used object detection models such as R-CNN, customized existing models, and build own models to detect, localize, and label rubber duck images
	- Implemented image segmentation using variations of the fully convolutional network (FCN) including U-Net and Mask-RCNN to identify and detect numbers, pets, zombies
	- Identified which parts of an image are being used by the model to make its predictions using class activation maps and saliency maps

- [Course4: Generative Deep Learning with TensorFlow](https://www.coursera.org/learn/generative-deep-learning-with-tensorflow?specialization=tensorflow-advanced-techniques)
	- Generated artwork using neural style transfer: extract the content of an image (eg. swan), and the style of a painting (eg. cubist or impressionist), and combine the content and style into a new image
	- Build simple AutoEncoders on the familiar MNIST dataset, and more complex deep and convolutional architectures on the Fashion MNIST dataset
	- Identified ways to de-noise noisy images, and build a CNN AutoEncoder using TensorFlow to output a clean image from a noisy one
	- Build Variational AutoEncoders (VAEs) to generate entirely new data, and generated anime faces to compare them against reference images
	- Learned about GANs, the concept of 2 training phases, the role of introduced noise and build GANs to generate faces


## Programming Assignments<a name="Programming"></a>
1. **[Course 1: Custom Models, Layers, and Loss Functions with TensorFlow](https://github.com/Ankit-Kumar-Saini/Coursera_TensorFlow_Advanced_Techniques_Specialization/tree/main/C1%20-%20Custom%20Models%2C%20Layers%2C%20and%20Loss%20Functions%20with%20TensorFlow)**

	- **[Week 1: Multiple Output Models using the Keras Functional API](https://github.com/Ankit-Kumar-Saini/Coursera_TensorFlow_Advanced_Techniques_Specialization/blob/main/C1%20-%20Custom%20Models%2C%20Layers%2C%20and%20Loss%20Functions%20with%20TensorFlow/Week1/C1W1_Assignment.ipynb)**

	- **[Week 2: Creating a Custom Loss Function](https://github.com/Ankit-Kumar-Saini/Coursera_TensorFlow_Advanced_Techniques_Specialization/blob/main/C1%20-%20Custom%20Models%2C%20Layers%2C%20and%20Loss%20Functions%20with%20TensorFlow/Week2/C1W2_Assignment.ipynb)**

	- **[Week 3: Implement a Quadratic Layer](https://github.com/Ankit-Kumar-Saini/Coursera_TensorFlow_Advanced_Techniques_Specialization/tree/main/C1%20-%20Custom%20Models%2C%20Layers%2C%20and%20Loss%20Functions%20with%20TensorFlow/Week3)**

	- **[Week 4: Create a VGG network](https://github.com/Ankit-Kumar-Saini/Coursera_TensorFlow_Advanced_Techniques_Specialization/blob/main/C1%20-%20Custom%20Models%2C%20Layers%2C%20and%20Loss%20Functions%20with%20TensorFlow/Week4/C1W4_Assignment.ipynb)**

	- **[Week 5: Introduction to Keras callbacks](https://github.com/Ankit-Kumar-Saini/Coursera_TensorFlow_Advanced_Techniques_Specialization/blob/main/C1%20-%20Custom%20Models%2C%20Layers%2C%20and%20Loss%20Functions%20with%20TensorFlow/Week5/C1_W5_Lab_1_exploring-callbacks.ipynb)**


2. **[Course 2: Custom and Distributed Training with TensorFlow](https://github.com/Ankit-Kumar-Saini/Coursera_TensorFlow_Advanced_Techniques_Specialization/tree/main/C2%20-%20Custom%20and%20Distributed%20Training%20with%20TensorFlow)**

	- **[Week 1: Basic Tensor operations and GradientTape](https://github.com/Ankit-Kumar-Saini/Coursera_TensorFlow_Advanced_Techniques_Specialization/blob/main/C2%20-%20Custom%20and%20Distributed%20Training%20with%20TensorFlow/Week1/C2W1_Assignment.ipynb)**

	- **[Week 2: Breast Cancer Prediction](https://github.com/Ankit-Kumar-Saini/Coursera_TensorFlow_Advanced_Techniques_Specialization/blob/main/C2%20-%20Custom%20and%20Distributed%20Training%20with%20TensorFlow/Week2/C2W2_Assignment.ipynb)**

	- **[Week 3: Horse or Human? In-graph training loop](https://github.com/Ankit-Kumar-Saini/Coursera_TensorFlow_Advanced_Techniques_Specialization/blob/main/C2%20-%20Custom%20and%20Distributed%20Training%20with%20TensorFlow/Week3/C2W3_Assignment.ipynb)**

	- **[Week 4: Custom training with tf.distribute.Strategy](https://github.com/Ankit-Kumar-Saini/Coursera_TensorFlow_Advanced_Techniques_Specialization/blob/main/C2%20-%20Custom%20and%20Distributed%20Training%20with%20TensorFlow/Week4/C2W4_Assignment.ipynb)**
  

3. **[Course 3: Advanced Computer Vision with TensorFlow](https://github.com/Ankit-Kumar-Saini/Coursera_TensorFlow_Advanced_Techniques_Specialization/tree/main/C3%20-%20Advanced%20Computer%20Vision%20with%20TensorFlow)**

	- **[Week 1: Predicting Bounding Boxes](https://github.com/Ankit-Kumar-Saini/Coursera_TensorFlow_Advanced_Techniques_Specialization/blob/main/C3%20-%20Advanced%20Computer%20Vision%20with%20TensorFlow/Week1/C3W1_Assignment.ipynb)**
    
	- **[Week 2: Zombie Detection](https://github.com/Ankit-Kumar-Saini/Coursera_TensorFlow_Advanced_Techniques_Specialization/blob/main/C3%20-%20Advanced%20Computer%20Vision%20with%20TensorFlow/Week2/C3W2_Assignment.ipynb)**

	- **[Week 3: Image Segmentation of Handwritten Digits](https://github.com/Ankit-Kumar-Saini/Coursera_TensorFlow_Advanced_Techniques_Specialization/blob/main/C3%20-%20Advanced%20Computer%20Vision%20with%20TensorFlow/Week3/C3W3_Assignment.ipynb)**

	- **[Week 4: Saliency Maps](https://github.com/Ankit-Kumar-Saini/Coursera_TensorFlow_Advanced_Techniques_Specialization/blob/main/C3%20-%20Advanced%20Computer%20Vision%20with%20TensorFlow/Week4/C3W4_Assignment.ipynb)**
   

4. **[Course 4: Generative Deep Learning with TensorFlow](https://github.com/Ankit-Kumar-Saini/Coursera_TensorFlow_Advanced_Techniques_Specialization/tree/main/C4%20-%20Generative%20Deep%20Learning%20with%20TensorFlow)**

	- **[Week 1: Neural Style Transfer](https://github.com/Ankit-Kumar-Saini/Coursera_TensorFlow_Advanced_Techniques_Specialization/blob/main/C4%20-%20Generative%20Deep%20Learning%20with%20TensorFlow/Week1/C4W1_Assignment.ipynb)**
    
	- **[Week 2: CIFAR-10 Autoencoder](https://github.com/Ankit-Kumar-Saini/Coursera_TensorFlow_Advanced_Techniques_Specialization/blob/main/C4%20-%20Generative%20Deep%20Learning%20with%20TensorFlow/Week2/C4W2_Assignment.ipynb)**

	- **[Week 3: Variational Autoencoders on Anime Faces](https://github.com/Ankit-Kumar-Saini/Coursera_TensorFlow_Advanced_Techniques_Specialization/blob/main/C4%20-%20Generative%20Deep%20Learning%20with%20TensorFlow/Week3/C4W3_Assignment.ipynb)**

	- **[Week 4: GANs with Hands](https://github.com/Ankit-Kumar-Saini/Coursera_TensorFlow_Advanced_Techniques_Specialization/blob/main/C4%20-%20Generative%20Deep%20Learning%20with%20TensorFlow/Week4/C4W4_Assignment.ipynb)**

## Results<a name="Result"></a>
`Some results from the programming assignments of this specialization`

1. Object detection on MNIST digits dataset
![alt text](https://github.com/Ankit-Kumar-Saini/Coursera_TensorFlow_Advanced_Techniques_Specialization/blob/main/sample_images/digit_detection.PNG)


2. Object detection using RetinaNet
![alt text](https://github.com/Ankit-Kumar-Saini/Coursera_TensorFlow_Advanced_Techniques_Specialization/blob/main/sample_images/retina_net.PNG)


3. Object detection using Mask RCNN
![alt text](https://github.com/Ankit-Kumar-Saini/Coursera_TensorFlow_Advanced_Techniques_Specialization/blob/main/sample_images/mask_rcnn.PNG)


4. Image segmentation on MNIST digits dataset
![alt text](https://github.com/Ankit-Kumar-Saini/Coursera_TensorFlow_Advanced_Techniques_Specialization/blob/main/sample_images/digits_segmentation.PNG)


5. Image segmentation on pets dataset
![alt text](https://github.com/Ankit-Kumar-Saini/Coursera_TensorFlow_Advanced_Techniques_Specialization/blob/main/sample_images/image_segmentation.PNG)


6. Scene segmentation
![alt text](https://github.com/Ankit-Kumar-Saini/Coursera_TensorFlow_Advanced_Techniques_Specialization/blob/main/sample_images/scene_segmentation.PNG)


7. Saliency maps
![alt text](https://github.com/Ankit-Kumar-Saini/Coursera_TensorFlow_Advanced_Techniques_Specialization/blob/main/sample_images/saliency_maps.PNG)