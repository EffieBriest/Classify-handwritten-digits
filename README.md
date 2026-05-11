# Handwritten Digit Classification with a Convolutional Neural Network
*Author: 吳斐力 (Felix Uhl)*

## Convolutional Neural Networks

Convolutional neural networks (CNNs) are neural networks designed for data with spatial structure, such as images. Instead of treating an image as an unstructured vector of pixels, a CNN preserves local spatial relationships and learns features through convolutional filters.

For image classification, this is useful because relevant patterns often appear locally. Simple structures such as edges, curves, and corners can be detected in early layers and combined into more complex representations in deeper layers.

A typical CNN consists of three main parts:

     convolutional layers    -> feature extraction
     pooling layers          -> spatial reduction
     fully connected layers  -> classification

## Image Representation

A grayscale image can be represented as a matrix

$$
X \in \mathbb{R}^{H \times W},
$$

where $H$ denotes the image height and $W$ denotes the image width.

More generally, an image with multiple channels is represented as a tensor

$$
X \in \mathbb{R}^{C_{\text{in}} \times H \times W},
$$

where $C_{\text{in}}$ is the number of input channels. For grayscale images, $C_{\text{in}} = 1$.

The goal of the network is to transform the input tensor into a feature representation that is useful for classification.

## Convolutional Layers

A convolutional layer applies learnable filters, also called kernels, to local regions of the input. For a kernel size of $k \times k$, one filter is given by

$$
K \in \mathbb{R}^{C_{\text{in}} \times k \times k}.
$$

At each spatial position, the convolution computes a weighted sum over a local region of the input. For one output feature map, this can be written as

$$
Y_{i,j}=\sum_{c=1}^{C_{\text{in}}}\sum_{m=0}^{k-1}\sum_{n=0}^{k-1}K_{c,m,n} X_{c,i+m,j+n}+ b,
$$

where $b$ is a learnable bias term.

The resulting matrix $Y$ is called a feature map. It describes how strongly the filter responds at each spatial position.

![convolution](https://hackmd.io/_uploads/r1BAD100Ze.jpg)

A convolutional layer usually learns several filters at once. If the layer has $C_{\text{out}}$ filters, it produces $C_{\text{out}}$ feature maps:

$$
Y \in \mathbb{R}^{C_{\text{out}} \times H' \times W'}.
$$

The dimensions $H'$ and $W'$ depend on the input size, kernel size, stride, and padding.

The filters are not manually designed. They are initialized automatically and optimized during training through backpropagation. A central property of convolutional layers is weight sharing: the same filter is applied at every spatial position. This reduces the number of parameters and allows the same pattern to be detected in different parts of the image.

## Activation Function

After a convolutional layer, a nonlinear activation function is applied. A common choice is the ReLU function:

$$
ReLU(x) = \max(0,x).
$$

ReLU is applied element-wise. Negative values are set to zero, while positive values remain unchanged.

The output of a convolutional filter can be interpreted as a feature response. A positive value indicates that the corresponding local region provides evidence for the pattern represented by the filter. A negative value indicates that this specific pattern is not active at that position, or that an opposite pattern is present.

Thus, ReLU creates a sparse activation map:

     0         -> feature inactive
     positive  -> feature active



## Pooling

Pooling reduces the spatial size of feature maps. A common choice is max pooling. For a pooling window of size $r \times r$, max pooling is defined by

$$
Y_{i,j}=
\max_{0 \leq m,n < r}
X_{ri+m,rj+n}.
$$

For example, a $2 \times 2$ max pooling operation maps

     [[1, 3],
      [2, 4]]

to

     4
![maxpooling](https://hackmd.io/_uploads/ByfzrbJ1Mx.png)
Max pooling keeps the strongest activation in each local region. This reduces computational cost and makes the representation less sensitive to small spatial shifts in the input.

## Flattening

After the convolutional and pooling layers, the output is still a tensor. For one input, it can be written as

$$
X \in \mathbb{R}^{C \times H \times W},
$$

where $C$ is the number of feature maps and $H,W$ are their spatial dimensions.

Before passing this representation into fully connected layers, it is reshaped into a vector:

$$
vec: \mathbb{R}^{C \times H \times W}
\rightarrow
\mathbb{R}^{CHW}.
$$

Thus,

$$
x = vec(X) \in \mathbb{R}^{CHW}.
$$

Flattening introduces no trainable parameters. It only changes the shape of the data. For a batch of $B$ inputs, the shape changes from

$$
B \times C \times H \times W
$$

to

$$
B \times CHW.
$$

## Fully Connected Layers

The flattened feature vector is passed through one or more fully connected layers. A fully connected layer applies an affine transformation

$$
z = Wx + b,
$$

where

$$
x \in \mathbb{R}^{n},
\qquad
W \in \mathbb{R}^{m \times n},
\qquad
b \in \mathbb{R}^{m},
\qquad
z \in \mathbb{R}^{m}.
$$

In a CNN, the convolutional layers extract features, while the fully connected layers use these features to compute class scores.

A classifier with one hidden layer can be written as

$$
h = \sigma(W_1x + b_1),
$$

$$
z = W_2h + b_2,
$$

where $\sigma$ is a nonlinear activation function such as ReLU. The final vector $z$ contains the raw class scores of the model.

## Logits and Softmax

For a classification problem with $C$ classes, the final layer outputs a vector

$$
z = (z_1, z_2, \dots, z_C) \in \mathbb{R}^{C}.
$$

The entries $z_i$ are called logits. They are unnormalized class scores and are not probabilities.

To convert logits into a probability distribution, the softmax function is used:

$$
p_i =
\frac{e^{z_i}}
{\sum_{j=1}^{C} e^{z_j}},
\qquad i = 1,\dots,C.
$$

The resulting vector

$$
p = (p_1, p_2, \dots, p_C)
$$

satisfies

$$
p_i \geq 0
$$

and

$$
\sum_{i=1}^{C} p_i = 1.
$$

The predicted class is chosen by selecting the largest logit, equivalently the largest softmax probability:

$$
\hat{y}=
\arg\max_i z_i=
\arg\max_i p_i.
$$

## Loss Function

For multi-class classification, the model is trained using cross-entropy loss. Let

$$
z = (z_1, z_2, \dots, z_C) \in \mathbb{R}^C
$$

be the logits produced by the model for $C$ classes. Applying the softmax function gives the predicted probability distribution

$$
p_i =
\frac{e^{z_i}}
{\sum_{j=1}^{C} e^{z_j}},
\qquad i = 1,\dots,C.
$$

If the true class is

$$
y \in \{1,2,\dots,C\},
$$

then the cross-entropy loss for one training example is

$$
\mathcal{L}(y,p)=
-\log(p_y),
$$

where $p_y$ is the probability assigned to the correct class.

Thus, the loss penalizes the model when it assigns low probability to the correct class. 

## Adam Optimizer

After the loss is computed, the model parameters are updated using the Adam optimizer. Adam stands for *Adaptive Moment Estimation*. It combines momentum-based updates with adaptive learning rates for each trainable parameter.

Let

$$
\theta_t
$$

denote the model parameters at training step $t$, and let

$$
g_t = \nabla_{\theta}\mathcal{L}_t
$$

be the gradient of the loss with respect to these parameters.

Adam stores two moving averages for every parameter. The first moment estimates the average gradient direction:

$$
m_t =
\beta_1 m_{t-1}
+
(1-\beta_1)g_t.
$$

The second moment estimates the average squared gradient:

$$
v_t =
\beta_2 v_{t-1}
+
(1-\beta_2)g_t^2.
$$

The square $g_t^2$ is computed element-wise. The constants $\beta_1$ and $\beta_2$ control how much past gradients influence the current update. Common default values are

$$
\beta_1 = 0.9,
\qquad
\beta_2 = 0.999.
$$

Since both moving averages are initialized at zero, they are biased toward zero during the first training steps. Adam corrects this bias by computing

$$
\hat{m}_t =
\frac{m_t}{1-\beta_1^t},
$$

and

$$
\hat{v}_t =
\frac{v_t}{1-\beta_2^t}.
$$

The parameter update is then

$$
\theta_{t+1}=\theta_t-\eta
\frac{\hat{m}_t}
{\sqrt{\hat{v}_t}+\varepsilon},
$$

where $\eta$ is the learning rate and $\varepsilon$ is a small constant for numerical stability.

The first moment $\hat{m}_t$ smooths the update direction, similar to momentum. The second moment $\hat{v}_t$ rescales the update for each parameter individually. Parameters with consistently large gradients receive smaller effective updates, while parameters with smaller gradients can receive relatively larger updates.

In a convolutional neural network, the parameter vector $\theta$ includes the convolutional filters, convolutional biases, and the weights and biases of the fully connected layers. For a convolutional filter

$$
K \in \mathbb{R}^{C_{\text{in}} \times k \times k},
$$

each entry $K_{c,m,n}$ is a trainable parameter. During backpropagation, the gradient

$$
\frac{\partial \mathcal{L}}{\partial K_{c,m,n}}
$$

is computed for every filter entry. Adam then updates each entry separately:

$$
K_{t+1,c,m,n}=K_{t,c,m,n}-
\eta
\frac{\hat{m}_{t,c,m,n}}
{\sqrt{\hat{v}_{t,c,m,n}}+\varepsilon}.
$$

Thus, Adam does not update a convolutional filter only as one complete matrix. Instead, every value inside the filter receives its own adaptive update.

As a simple example, consider a $2 \times 2$ filter

$$
K_t =
\begin{bmatrix}
0.20 & -0.10 \\
0.05 & 0.30
\end{bmatrix}.
$$

After computing the cross-entropy loss and applying backpropagation, suppose the gradient with respect to this filter is

$$
g_t =
\frac{\partial \mathcal{L}}{\partial K_t}=
\begin{bmatrix}
0.04 & -0.02 \\
0.01 & 0.08
\end{bmatrix}.
$$

A positive gradient means that increasing the corresponding filter value would increase the loss, so the optimizer should decrease that value. A negative gradient means that increasing the value would decrease the loss, so the optimizer should increase it.

For the first Adam update, assume

$$
m_0 = 0,
\qquad
v_0 = 0,
$$

with

$$
\beta_1 = 0.9,
\qquad
\beta_2 = 0.999.
$$

The first moment is

$$
m_1=
0.9m_0 + 0.1g_t=
0.1g_t=
\begin{bmatrix}
0.004 & -0.002 \\
0.001 & 0.008
\end{bmatrix},
$$

and the second moment is

$$
v_1=
0.999v_0 + 0.001g_t^2=
0.001g_t^2=
\begin{bmatrix}
0.0000016 & 0.0000004 \\
0.0000001 & 0.0000064
\end{bmatrix}.
$$

After bias correction,

$$
\hat{m}_1 = g_t,
\qquad
\hat{v}_1 = g_t^2.
$$

With learning rate $\eta = 0.001$, the update is

$$
K_{t+1}=K_t-
\eta
\frac{\hat{m}_1}
{\sqrt{\hat{v}_1}+\varepsilon}.
$$

Ignoring the very small $\varepsilon$ for readability, this gives approximately

$$
K_{t+1}\approx\begin{bmatrix}
0.20 & -0.10 \\
0.05 & 0.30
\end{bmatrix}-
0.001
\begin{bmatrix}
1 & -1 \\
1 & 1
\end{bmatrix}.
$$

Therefore,

$$
K_{t+1}
\approx
\begin{bmatrix}
0.199 & -0.099 \\
0.049 & 0.299
\end{bmatrix}.
$$

Repeating this process over many training steps gradually changes the initially random filters into filters that respond to useful visual structures in the input data.

# Implementation with PyTorch

After establishing the theoretical foundation of convolutional neural networks, this section describes how the digit classifier is implemented in PyTorch. The implementation follows a standard image classification pipeline: data loading, model definition, training, evaluation, model saving, model loading, and prediction on new images.

---

## PyTorch

PyTorch is a Python library for building and training neural networks. It provides tensor operations, automatic differentiation, model building blocks, loss functions, and optimizers.

In this project, PyTorch is used to define the convolutional neural network, train the model on MNIST images, calculate the loss, update the model parameters, and evaluate the model performance.


## Data Preparation

The MNIST dataset is loaded using torchvision:

     datasets.MNIST(
         root="./data",
         train=True,
         download=True,
         transform=transforms.ToTensor()
     )

The argument

     root="./data"

defines where the dataset is stored locally. If the dataset does not already exist, it is downloaded automatically because

     download=True

is used.

The transformation

     transforms.ToTensor()

converts each image into a PyTorch tensor and scales the pixel values from the range `0–255` to the range `0–1`.

Each image has the shape:

     [1, 28, 28]

where:

* `1` is the grayscale channel,
* `28` is the image height,
* `28` is the image width.

When images are processed in batches, the shape becomes:

     [batch_size, 1, 28, 28]

The corresponding label is an integer between `0` and `9`.

---

## Model Architecture

The model is implemented as a PyTorch class that inherits from `nn.Module`. It consists of two convolutional blocks followed by a linear classifier.

The input images are grayscale MNIST images with shape

$$
1 \times 28 \times 28.
$$

The model uses a hidden dimension of

$$
64,
$$

meaning that the convolutional layers learn 64 feature maps.

The first convolutional block receives the original grayscale image and applies two convolutional layers with ReLU activations, followed by max pooling:

     Conv2d: 1 -> 64 feature maps
     ReLU
     Conv2d: 64 -> 64 feature maps
     ReLU
     Max pooling

Both convolutional layers use a kernel size of $3 \times 3$, stride $1$, and padding $1$. Because of this padding, the convolutional layers preserve the spatial size:

     28 x 28 -> 28 x 28

The max pooling layer then reduces the spatial size by a factor of two:

     28 x 28 -> 14 x 14

After the first block, the tensor has shape

$$
64 \times 14 \times 14.
$$

The second convolutional block has the same structure, but it receives 64 input feature maps instead of one grayscale channel:

     Conv2d: 64 -> 64 feature maps
     ReLU
     Conv2d: 64 -> 64 feature maps
     ReLU
     Max pooling

Again, the convolutional layers preserve the spatial dimensions, while max pooling reduces them:

     14 x 14 -> 7 x 7

After the second convolutional block, the tensor has shape

$$
64 \times 7 \times 7.
$$

Before classification, this tensor is flattened into a vector:

$$
64 \cdot 7 \cdot 7 = 3136.
$$

The classifier then maps this feature vector directly to the output classes:

     Flatten
     Linear: 3136 -> 10

The final layer outputs 10 logits, one for each digit class.

The complete architecture can be summarized as:

     Input image
     -> convolutional block 1
     -> convolutional block 2
     -> flattening
     -> linear classifier
     -> logits

A notable aspect of this architecture is that the classifier contains only one linear layer. Most of the learning therefore happens inside the convolutional blocks, which extract visual features from the image. The final linear layer only maps these extracted features to the 10 output classes.

---

## Training Process

The model is trained on the MNIST dataset using mini-batches of size $32$:

```python
train_loader = load_mnist(32, training_mode=True)
test_loader = load_mnist(32, training_mode=False)
```

The model is initialized with $10$ output classes, corresponding to the digits from $0$ to $9$:

```python
model = DigitRecognizitonModel(10)
```

For multi-class classification, the loss function is cross-entropy loss:

```python
loss_fn = nn.CrossEntropyLoss()
```

The model outputs raw logits, not probabilities. These logits are passed directly into `CrossEntropyLoss`, which internally applies the softmax operation in a numerically stable way. Therefore, no separate softmax layer is needed during training.

The optimizer used is Adam with learning rate

$$
\eta = 0.001.
$$

```python
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
```

Adam updates all trainable parameters of the model, including the convolutional filters, convolutional biases, and the weights and biases of the final linear layer.

The model is trained for $3$ epochs:

```python
epochs = 3
```

During training, the model is set to training mode:

```python
model.train()
```

For each batch, the input images are passed through the network, the loss is computed, and the parameters are updated:

```python
y_pred = model(X)
loss = loss_fn(y_pred, y)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

This is the central training step. First, the model computes the logits for the current batch. Then the cross-entropy loss compares these logits with the true labels. Before backpropagation, the old gradients are cleared. The call to `loss.backward()` computes the new gradients, and `optimizer.step()` updates the model parameters using Adam.

The training loss is accumulated over all batches and averaged at the end of each epoch.

After each training epoch, the model is evaluated on the test dataset. During evaluation, the model is set to evaluation mode and gradient computation is disabled:

```python
model.eval()

with torch.inference_mode():
    ...
```

This is important because no parameters should be updated during testing. Disabling gradient computation also makes evaluation faster and more memory-efficient.

For each test batch, the model again outputs logits. The predicted class is obtained by selecting the index with the largest logit:

```python
test_pred.argmax(dim=1)
```

Mathematically, this corresponds to

$$
\hat{y} = \arg\max_i z_i.
$$

The test loss and test accuracy are accumulated over all test batches and averaged at the end of the epoch. The training process therefore clearly separates two phases:

     training   -> compute gradients and update parameters
     evaluation -> measure performance on unseen data

## Training Process

The model is trained using the standard PyTorch training workflow.

For each batch:

1. The images are passed through the model.
2. The model outputs logits for the 10 digit classes.
3. The loss is calculated using cross-entropy loss.
4. The optimizer clears the old gradients.
5. Backpropagation computes the new gradients.
6. The optimizer updates the model parameters.
7. The predictions are compared with the true labels to calculate accuracy.

The loss function is:

     nn.CrossEntropyLoss()

The optimizer used is Adam:

     torch.optim.Adam(model.parameters(), lr=0.001)

The model is trained for a fixed number of epochs. During training, the training loss, training accuracy, test loss, and test accuracy can be recorded and visualized.


## Model Evaluation

After training, the model is evaluated on the test dataset.

Evaluation is done with:

     model.eval()

and

     torch.inference_mode()

This ensures that the model is used only for prediction and that no gradients are calculated.

The predicted class is obtained with:

     y_pred = torch.argmax(logits, dim=1)

For example, if the model outputs:

     [0.1, -1.2, 0.4, 2.7, -0.5, 0.2, -1.0, 0.8, 0.3, -0.6]

then the largest value is at index `3`, so the model predicts:

     Prediction: 3

The prediction is correct if this value matches the true label.

---

## Prediction on New Images

After training, the model can be used to classify new digit images.

However, the image must be prepared in the same format as the MNIST training data.

The model expects:

     28 x 28 pixels
     grayscale
     one centered digit
     similar style to MNIST
     tensor shape: [1, 1, 28, 28]

A custom image should therefore be converted to grayscale, resized to `28 x 28`, transformed into a tensor, and passed through the model.

## Model Saving and Loading

After training, the user will be ask if he wants to safe the model: 

![Save Model](https://hackmd.io/_uploads/r180300A-l.png)

If the model is to be safed, the user needs to enter a model name. 

![save model 2](https://hackmd.io/_uploads/H1A2pCAAZg.png)

".pth" is not needed to be added.


The trained model parameters are saved in a `.pth` file:

     digit_model.pth

This allows the model to be loaded again without retraining.

To load the model again, the same model architecture must first be created. Then the saved parameters can be loaded:

     model = DigitRecognitionModel()
     model.load_state_dict(torch.load("models/digit_model.pth"))
     model.eval()

The model architecture and the saved weights must match. If the architecture is changed, the old `.pth` file may no longer load correctly.

## Error Analysis

After evaluation, incorrect predictions can be inspected manually.

The model predictions are compared with the true labels. Cases where the prediction does not match the label are extracted:

     correct = prediction == true_label

The 16 of the incorrect examples can then be visualized together with their true and predicted labels.

For example:

![error analysis](https://hackmd.io/_uploads/H1vg0RAAWg.png)


Error analysis helps identify where the model struggles. Common causes of mistakes may include:

* unusual handwriting,
* digits that look similar, such as `4` and `9`,
* digits that are not centered,
* very thin or unclear strokes,

Accuracy alone does not explain why a model makes mistakes. Looking at incorrect predictions gives a more practical understanding of the model’s behavior.

## Results

The convolutional neural network performs well on the MNIST dataset. Since MNIST is a clean and commonly used benchmark dataset, even a relatively small CNN can achieve high test accuracy.

Interesting parameters to compare are:

* number of epochs,
* learning rate,
* batch size,
* number of convolutional layers,
* number of filters,
* kernel size,
* optimizer choice.

### Epochs

Training for more epochs generally improves the model at the beginning. However, after a certain point, the improvement becomes smaller. If the model is trained for too long, it may start to overfit the training data. For this specific MNIST setup, one epoch was already enough to achieve strong performance. Additional epochs provided only small improvements.

![epochs](https://hackmd.io/_uploads/rkvxXJk1fx.png)

### Learning Rate

The learning rate controls how strongly the model parameters are updated during training.

If the learning rate is too large, training may become unstable and the model may not converge properly. If the learning rate is too small, training can become very slow.

![learningrate compare](https://hackmd.io/_uploads/B1ZXlZkkGe.jpg)


### Batch Size

Changing the batch size had only a minor effect on the final model accuracy. This is expected for a relatively simple and well-structured dataset such as MNIST, where the model can learn stable visual patterns across a range of batch sizes.

Smaller batch sizes produce more frequent but noisier gradient updates, while larger batch sizes provide smoother gradient estimates. In this experiment, however, these differences mainly affected the training dynamics rather than the final classification performance.
![batchsize comparison](https://hackmd.io/_uploads/SkSJW-1kGg.jpg)


### Model Architecture

The number of convolutional layers and filters influences how many visual patterns the model can learn.

A very small model may underfit because it cannot learn enough useful features. A very large model may be unnecessary for MNIST and could overfit or require more training time. In this specific case, 2 convolution blocks and 3 convolution blocks are producing the same result. 
![conv blocks comparison](https://hackmd.io/_uploads/HkDCrZy1Mx.jpg)


## Conclusion

This project demonstrates how convolutional neural networks can be used for handwritten digit classification. The model learns visual features directly from image data and uses these features to classify digits from `0` to `9`.

Compared to a simple fully connected network, a CNN is better suited for image data because it preserves spatial structure and learns local patterns such as edges, curves, and digit parts.

Although the model performs well on MNIST, it is mainly suitable for MNIST-like handwritten digits. For real-world digit recognition, the dataset and preprocessing pipeline would need to be extended.
--
## References
* GeeksforGeeks [link](https://www.geeksforgeeks.org/machine-learning/introduction-convolution-neural-network/)
* Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization*. International Conference on Learning Representations (ICLR). [link](https://arxiv.org/abs/1412.6980)
* O'Shea, K., & Nash, R. (2015). *An Introduction to Convolutional Neural Networks*. arXiv. [link](https://arxiv.org/abs/1511.08458)
