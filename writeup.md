# Project: Follow me

[nn-softmax]: ./writeup-material/nn-softmax.png
[nn-ce]: ./writeup-material/nn-ce.png
[nn-gradient-opt]: ./writeup-material/nn-gradient-opt.png
[dnn]: ./writeup-material/dnn.png
[cnn]: ./writeup-material/cnn.jpg
[cnn-max-pool]: ./writeup-material/cnn-max-pool.png
[fcn]: ./writeup-material/fcn.png
[fcn-separable]: ./writeup-material/fcn-separable.png
[fcn-iou]: ./writeup-material/fcn-iou.png
[train]: ./writeup-material/train.png
[follow_target]: ./writeup-material/follow_target.png
[with_target]: ./writeup-material/with_target.png
[without_target]: ./writeup-material/without_target.png

---

## 1. Deep Learning

Before, artificial intellegce is achieved by filtering targets with lots of human-made rules. This is not robust. Recently, deep learning has a great advance, because of the performance improvement of computing components, especially GPU (Graphics processing unit). This technology takes a lot of data, and tries to find models automatically, without human-made rules, to predict with high accuracy. It successfully make great contributions to fields like: image recognitions, illness diagnosis, self-driving car, etc.


## 2. NN (Neural Network)

The building block of deep learning is a perceptron. It is similar to biology neuron operations.

For standard problems, we will have input `X`, and ground truth output `y`. First, a bias `b` is added with weighted inputs `X * W`, and the output is logits `y_bar`. Using a __softmax function, predicted logits are converted to probabilities `y_hat`__. 

![alt text][nn-softmax]

An error function tells the similarity of two vectors, and we can compare logits `y_hat` and labels `y` with this function, because idea logits should be the same as the labels. For an error function, we adopt a __cross-entropy function, and compares the likelyhood between prediction probabilities and the ground truth__. 

Naively, if we want to maxmize all correct probabilities, we can rely on products of all probabilities. However, this large production is computation ineffective, and sensitive to small errors. Therefore, in the cross-entropy function, `log` functions convert multiplications to additions. Since probabilities always in the range from 0 to 1, and its `log` is a negative value. An additional minus sign is placed to ensure a postive value. `-ln(0.1)` is `2.302`, and `-ln(0.9)` is `0.105`. Misclassified components have lower probabilities, and result in big cross-entropy contributions. `Loss` is the result of dividing cross entropy to the number of dataset.

![alt text][nn-ce]

 Meanwhile, the gradient of the loss function reveals the direction to minimize errors. In the end, the optimization is made by adjusting weight and bias according to the loss gradient and learning rate. By iterating the above process, we can approcah the model to achieve higher accuracy.

![alt text][nn-gradient-opt]

__Split dataset into training / validation / test__: During training, we use a validation set to check the performance of NN. As iterations, the model will also memorize, not learn, the behavior of the validation set. We need an additioal test set, which haven't be seen by the model in the training process. It will be used in the final performance verification.


## 3. DNN (Deep Neural Network)

#### 3.1. DNN = add non-linearities to NN

Although a NN can solve a linear-classification problem, it performs bad on non-linear functions. Deep neural network overcomes this defficiency by adding stacks of neural networks and non-linear activation functions together, and this __introduces non-linearities into the model__. As a result, this is why we call the technology as deep learning, since the structure containes several layers and is deep.

![alt text][dnn]


#### 3.2. SGD (Stochastic Gradient Descent)

When we take all data to compute the cross-entropy function, it will result in a hugh computation and requires a lot of memory. To avoid this scenario, we take a subset of data, called __batch__, and calculate graditent. To get a batch, we need to randomize data in advance, called __shuffle__, and it is a rough approximation to the whole dataset base on a statistic viewpoint. Then, the target can be found by iterating rough approximations through all datasets. This is named as SGD.


#### 3.3. Things to improve DNN/SGD

1. __Normalization__: We adjust input layer to zero mean, and small equal variance, and this can avoid the optimizer to search through all space.  

2. __Momentum__: This is introduced to use a running average of gradient, instead of using a direction of current batch.

3. __Learning rate decay__: Large learning rate results in a bounce forward and backward near the target, it may not converge to the target. On the other hand, it takes a long time to train with small learning rate. Learning rate decay can prevent these two downsides.


#### 3.4. Avoid Overfitting

When the model perform well on the training set, but bad on the test set, it is called overfitting. There are several techniques to prevent overfitting: 

- Early termination: Stop training when validation performance starts to decrease.
- L2 regularization: add additional term on the loss function to penalize hugh weights.
- Dropt out: Ramdomly remove input for each layers in the training. This can mitigate the network rely on any given activation.


## 4. CNN (Convolutional Neural Network)

If we want to detect whether a car is in an image, we don't care where a car is. This space statistical invariance can be achieved by CNN. Pixels are grouped as patches, and weighted together. The weight sharing mechanism is adopted to sweep the same weighting filter through images. Similar input patches will result in similar outputs. For a layer, we can sweep multiple weighting filters for different features. 

The following figure shows one layer CNN, where an input image is `(W=5, H=5, D=1)`; a filter is : `(W=3, H=3, D=3)`; stride is 1; padding is valid. The output feature is `(W=3, H=3, D=3)`.

![alt text][cnn]


__Max pooling__: A way to down sampling, and it can prevent overfitting. The following figure shows an example, which remove 75% of original pixels.

![alt text][cnn-max-pool]


## 5. FCN (Fully Convolutional Network)

To find object's locations in images, we can use the bounding box. It is fast, but it only yields partial scene understanding, and no true shape of objects can be retrieved. FCN provides a way to infer a true shape of objects in images. 

FCN contains an encoder and a decoder. The encoder is classical convolutional layers, while the decoder is transposed convolutional layers to upsampling. In between, there is a 1x1 convolutional layer to preserve spatial information.

In classical convolutional layers, layers identify local features, but lose big pictures. We need to add additional skip connection layers. They are element wise operations, and add information with multi resolutions.

![alt text][fcn]

Classical CNN connects convolution layers to fully connected layers. The output of fully connected layers are a 2D tensor: `[batch, labels]`, and spatial information is lost. We can connect convolution layers to __1x1 convolution layers__. They output 4D tensors: `[batch, # of filters, height, width]`, which keep the spatial information. Besides, it makes our architecture deeper, and the operation is cheap, becuase it is only matrix multiplications.

__Bilinear upsampling__: Upsampling with interpolation method. Distance-dependent weight-average of 4 nearest known pixels.

__Batch normalization__: Normalize inputs of each layers, not only the input layer. normalize using the current batch.

__Separable convolutional layer__: Reduce parameters for encoders

![alt text][fcn-separable]

__IOU (intersection over union metric):__

![alt text][fcn-iou]


## 6. Project: Follow me

[`model_training.ipynb` notebook](https://github.com/JasonYCHuang/RoboND-DeepLearning-Project/blob/master/result/model_training.ipynb)

[HTML version of notebook](https://github.com/JasonYCHuang/RoboND-DeepLearning-Project/blob/master/result/model_training.html)

[model and weights files](https://github.com/JasonYCHuang/RoboND-DeepLearning-Project/tree/master/result/model)

In the project, the encoder block is a separable convolution layer with batch normalization. On the other hand, the decoder block is a bilinear upsampling layer with skip connections. In between, there is a 1x1 convolution layer. In my model, both encoder and decoder have 3 layers respectively.

__Hyperparameters__ 

In order to increase IOU, I need to increase the capacity of training set to 6592. 

Larger `batch_size` can represent more infomation of the whole dataset. However, when I increase `batch_size` to 128, my GPU is running out of memory. Meanwhile, smaller `batch_size` will lead to a lower accuracy, since a small batch can not provide enough information linking to the whole dataset. I set 64 for `batch_size`.

__Results__ 

The training curve shows no sign of overfitting.

![alt text][train]

Images while following the target can be identified correctly, and there are some misclassified pixels around dark wall.

![alt text][follow_target]

Images while at patrol without target can be identified correctly, and there are some misclassified pixels around dark wall.

![alt text][without_target]

Images while at patrol with target show partial identifications, and target pixels are mixed with other people.

![alt text][with_target]

The final IOU is 0.425. 

In this project, I haven't implement momentum and learning rate decay. Moreover, filter sizes can be adjusted. These will be the future work.



