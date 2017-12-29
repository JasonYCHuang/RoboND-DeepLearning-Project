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
[follow_target]: ./writeup-material/follow_target.png
[with_target]: ./writeup-material/with_target.png
[without_target]: ./writeup-material/without_target.png
[model-structure]: ./writeup-material/model-structure.png
[curve]: ./writeup-material/curve.png

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

Classical CNN connects convolution layers to fully connected layers. The output of fully connected layers are a 2D tensor: `[batch, labels]`, and spatial information is lost. We can connect convolution layers to __1x1 convolution layers__. It outputs 4D tensors: `[batch, # of filters, height, width]`, which keep the spatial information. Meanwhile, we can modify filter space dimensionality. The coordinate-dependent transformation in the filter space is a cross-channel learning, and makes the model deeper[[1]](http://iamaaditya.github.io/2016/03/one-by-one-convolution/). The operation is cheap, becuase it is only matrix multiplications.

__Bilinear upsampling__: Upsampling with interpolation method. Distance-dependent weight-average of 4 nearest known pixels.

__Batch normalization__: Normalize inputs of each layers, not only the input layer. normalize using the current batch.

__Separable convolutional layer__: Reduce parameters for encoders

![alt text][fcn-separable]

__IOU (intersection over union metric):__

![alt text][fcn-iou]


## 6. Project: Follow me

[You can find notebook / html / model and weights files here.](https://github.com/JasonYCHuang/RoboND-DeepLearning-Project/tree/master/result)

#### 6.1. The student clearly explains each layer of the network architecture and the role that it plays in the overall network. The student can demonstrate the benefits and/or drawbacks of different network architectures pertaining to this project and can justify the current network with factual data. Any choice of configurable parameters should also be explained in the network architecture.

The following code and figure show the structure of the model.

```
def fcn_model(inputs, num_classes):
    
    # TODO Add Encoder Blocks. 
    ec_1 = encoder_block(inputs, 32, 2)
    ec_2 = encoder_block(ec_1, 48, 2)
    ec_3 = encoder_block(ec_2, 64, 2)
    
    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    conv_1x1 = conv2d_batchnorm(ec_3, 64, 1, 1)
    
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    dc_3 = decoder_block(conv_1x1, ec_2, 64)
    dc_2 = decoder_block(dc_3, ec_1, 48)
    dc_1 = decoder_block(dc_2, inputs, 32)
    
    x = dc_1
    
    # The function returns the output layer of your model. 
    # "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(x)
```

![alt text][model-structure]

__Encoder blocks (ec_1, ec_2, ec_3):__ Layers are constructed by separable convolution layers with batch normalizations. These are classical convolutional layers with modifications. Therefore, each layers can extract different features to learn target information. As layers go deeper, more complex features can be focused. Numbers of filters also increase as layers go deeper.

- Separable convolution layers decrease the number of parameters, and it also make training with less memory. A more detail explanation is in section 5.

- Batch normalizations reshape loss function to zero mean and small variance, which avoid search through all parameter space, and decrease time to reach the target.

__Decoder blocks (dc_1, dc_2, dc_3):__ These blocks restored original image spatial information, and integrate identified pixels for different objects and background. To expand compressed encoder blocks, layers are constructed by bilinear upsampling layers. Upsampling layers take 4 nearest known pixels, and calculate distance-dependent weight-average values base on the interpolation method. Numbers of filters also decrease as layers are close to the output.

Meanwhile, skip connections are added to add information with multi resolutions. They pass original higher level spatial information into the next layer which help the model using both local features from classical convolutional layers and global views from skip connections.


#### 6.2. The student is able to identify the use of various reasons for encoding / decoding images, when it should be used, why it is useful, and any problems that may arise.

In classical CNN, the model identify classes or targets by looking at specific features. As a result, classical CNN __converts input images into feature vectors__, and this is __encoding__.

On the other hand, __decoding__ is a process to __generate a semantic segmentation masek and restore feature vectors back to images with learned information__. In our case, the learned information is different pixels for target, other objects and background.

In layers, the `stride = 2` down-sampling results in the lose of information, but skip connections amends this. Moreover, the 1x1 convolution layer avoids spatial infomration lose when using a fully connected layer between encoder and decoder.


#### 6.3. The student demonstrates a clear understanding of 1 by 1 convolutions and where/when/how it should be used.

The 1 by 1 convolution outputs 4D tensors: `[batch, # of filters, height, width]`, which keeps the spatial information. Meanwhile, we can modify filter space dimensionality. This transformation in the filter space is a cross-channel learning, and makes the model deeper[[1]](http://iamaaditya.github.io/2016/03/one-by-one-convolution/). The operation is cheap, becuase it is only matrix multiplications.


#### 6.4. The student explains their neural network parameters including the values selected and how these values were obtained (i.e. how was hyper tuning performed? Brute force, etc.)

In order to increase IOU, I need to increase the capacity of training set to 6592. 

Larger `batch_size` can represent more infomation of the whole dataset. However, when I increase `batch_size` to 128, my GPU is running out of memory. Meanwhile, smaller `batch_size` will lead to a lower accuracy, since a small batch can not provide enough information linking to the whole dataset. I set 64 for `batch_size`.

`Learning rate` is base on previous exerience, and I didn't make more manual tunning since the final IOU reaches 40%. Besides, I also implement learning rate decay described in this post[[2]](https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/). This provides fast learning in the beginning because larger steps, and approaches the target with smaller steps in the later stage without bouncing back and forth around it because of large steps.

`epochs` is set to 100. In the beginning, I expected overfitting present when loss starts to increase. However, the curve only reaches a steady region. Therefore, I pick a model checkpoint with lowest loss.

![alt text][curve]

```
num_training_samples = 6592
num_valid_samples = 1185

batch_size = 64
num_epochs = 100
learning_rate = 0.1
decay_rate = learning_rate / num_epochs

steps_per_epoch = num_training_samples/batch_size
validation_steps = num_valid_samples/batch_size
workers = 4
```

The following line is modified to implement learning rate decay.

```
model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate, decay=decay_rate), loss='categorical_crossentropy')
```

#### 6.5. The student is able to clearly articulate whether this model and data would work well for following another object (dog, cat, car, etc.) instead of a human and if not, what changes would be required.

In the mask, we highlight target as blue, other objects as green and back ground as red. If we want to follow another project, we need to change blue pixels to the target which we want, and re-train the model parameters. This means the dataset needs to be modified. 

On the otherhand, The FCN structure could be kept as the same. Then, the re-train process will optimize weight and bias for the new target.

#### 6.6. Results

The training curve:

![alt text][curve]

Images while following the target can be identified correctly. In the target boundary, there are few pixels misclassified.

![alt text][follow_target]

Images while at patrol without target can be identified correctly, and there are some misclassified pixels around dark wall.

![alt text][without_target]

Images while at patrol with target show partial identifications.

![alt text][with_target]

The final IOU is 0.467. 


## Future Enhancements

- Implement momentum.

- Adjust filter sizes.

- Base on review feedbacks: The current training set is heavily biased towards background data. Creating an equal amount of background images, images with hero near and images with the hero away will solve this issue.


## Further reading

https://courses.cs.washington.edu/courses/cse576/17sp/notes/Sachin_Talk.pdf

https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/image_segmentation.html
