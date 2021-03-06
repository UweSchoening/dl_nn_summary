# Deep Learning and Neural Networks

---

    Disclaimer: No guarantee for correctnes or completeness. No official summary. Figures from the lecture. 

---

# 02/03 - Classification 1+2
First some primitive/ old methods for classification, then some based on NNs and Deep Learning (DL).

Overview:

![Overview](figures/01_11.PNG)


## Correlation-based => Template matching

### Cross Correlation
* 2 dimensional Cross-Correlation:

    $R_{(f,g)}(m, n) = \sum_i \sum_j f(i, j) * g(i-m, j-n)$
* 1 dimensional correlation respectively
* Autocorrelation = Cross correlation with itself

CC mostly is used to get a notion of normality.

### Template Matching

* Idea: Compare new patter to already known pattern, the template.
* **Similarity** $M_{(f,g)}(m,n)$ = Cross-Correlation between the two patterns m and n
* **Distance** E_{(f,g)}(m,n)= Square of Cross-correlation

=> Basically the same as folding a filter over a raster.

#### Problems
  * Cannot learn classification
  * Does not generalize well
  * Has to have a template for each class

#### Workarounds:
  * Normalisation
    * e.g. brightness
    * normalise axes through time non-linearly ("time warping")


## Parametric Classification
---

### Bayes Decision Theory

Datum $x$, class $w_j$

* A priori probability      $P(w_j)$
* A posteriori probability  $P(w_j|x)$
* Class-conditional probability $P(x|w_j)$

Bayes Rule: $P(w_j|x) = \frac{P(x|w_j) * P(w_j)}{P(x)}$

Decision rule: $P(w_j |x) > P(w_i|x)$ => choose $w_j$

$P(w_i | x) = \frac{P(x|w_i) * P(w_i)}{\sum_{j=1}^c P(x|w_j) * P(x)}$

=> decision: 

$P(x| w_j)*P(w_j) > P(x | w_i) * P(w_i)$ => $w_j$

    Common assumption: $p(x | w_i)$ is distributed according to a multivariate normal density

### Risk and Minimum Error Classification
* Choose class of lower risk <=> choose class that maximises posterior probability!

**Likelyhood Ratio**:

$\frac{p(x | w_q)}{p(x | w_2)} > \frac{\lambda_{12} - \lambda_{22}} {\lambda_{21} - \lambda_{11}} * \frac{P(w_1)}{P(w_2)}$

### Gaussian Mixture Classification

$\sum_i$ = covariance matrix

$\eta _i$ = mean vector

### Principal Component Analysis

### The curese of Dimensionality

## Non-parametric classification
---
* Parzen Window
* K-nearest neighbours
* Linear discriminants (general)
* Fisher-Linear Discriminant

### The Perceptron
![Perceptron](figures/04_06.png)

Classification: 

$g(x) < 0$ => Class A
$g(x) > 0$ => Not class A
$g(x) = 0$ => No decision

Where g(x) is:

$g(x) = \sum_{i=1}^n w_i*x_i + w_0$

with
* $x_i$ = feature i or 
  * $x$ = feature vector $x \in R^n$
* $w_i$ = weight of feature i or
  * $w$ = weight vector $w \in R^n$
* $w_o$ = **threshold**


## Unsupervised Classificatin/ Clustering
---

* Hierarchical Clustering
  * Choose number of classes ($n - c_{stop}$). initialize c = n = number of data points
  * Iteratively merge closest pairs of data points and decrement c

# 04 - Machine Learning 1 + 2

## Neural Networks
---

### General Information on NNs

* Advantages of Neural Networks
  * Massive parallelism
  * simple computing units
  * non-linear classifiers
  * massive contstraint satisfaction for ill-defined-input
  * uniformity
  * learning and adapting possible
  * learn hidden representations

* Design Criteria for NNs
  * performance
    * recognition error rate
    * generalisation
  * training time
  * recognition time
  * memory requirements
  * training complexitiy
  * ease of implementation
  * ease of adaption

* Applications of NNs
  * classification
  * prediction
  * function approximation
  * continuous mapping
  * pattern completion
  * coding
  * word/ feature embedding or encoding
  * encoding/ decoding

*  Network Specifications <br>
These parameters are usually chosen by the network architect
   * topology
   * node characteristics
   * learning rule
   * learning parameters
   * objective function
   * initial weights

* Design Problems of NNs
  * Local Minima
  * speed of learning
  * architecture must be selected
  * choice of feature representation
  * scaling
  * systems, modularity
  * treatment of temporal features and sequences

### Types of NNs
* MLPs
* Boltzman Machines
* Decision Tree Classifiers
* Feature Map Classifiers
* ART
* TDNNs / Convolutional Neural Networks

## Learning Techniques for NNs
Basically everything is Backpropagation (BP). However, there are some adaptions that make it more efficient.

### 1. Backpropagation

Each nueron computes $y = f(x)$ where
* $x_j = \sum_i y_i * w_{i,j}$  : imput to neuron j
* $y_j = f(x)$ with $f$ beeing an **activation function**, e.g. sigmoid:
  * **$f(x_j) = \frac{1}{1+\exp{(- x_j)}}$**
  * **$\partial f(x)/\partial x$ = f(x) * (1-f(x))**

**For classifiers the output $y_j$ of a neural network represents the  posteriori probability $p(w_j|x)$ for a class given some input data**

The Mean Square Error for ... <br>
...a neuron $j$ <br>
...with output $y_j$ <br>
...and target label (on that layer) $d_j$ is

$E(y_j, d_j) = 1/2 * \sum_j(y_j - d_j)$

---
BP approach:
  1. choose random initial weights
  2. apply input, get some output (Forward Pass)
  3. compare output to desired output and compute output error
  4. back-propagate error through network
     1. compute contribution of each weight to the overall output error:  $\frac{\partial E} {\partial w_{i,j}}$
     2. adjust weights slightly proportional to their share $\frac{\partial E} {\partial w_{i,j}}$ <br>
      $\Delta w_{ij}(t) = - \epsilon * \frac{\partial E} {\partial w_{i,j}}$ <br>
      $w_{ij}(t+1) = w_t + \Delta w_{ij}(t)$
---

### 2. Stochastic, Batch and Mini-Batch Gradient Descent
The critical point is, how much of the forward passed data shall be used to compute the gradient and to update a weight.
Meaning, shall an update of thei weights occur after each foward pass, after some (taking the average gradients) or after the whole data is classified/ passed forward once? There are also more elaborate methods like Adam and co.

For them look [here](https://ruder.io/optimizing-gradient-descent/).

In **stochastic gradient descent** **one** data sample is selected at **random** and proagated through the forward pass. After the propagation the gradient is computed. The computed gradient is then used to update the weights.

In **Batch Gradient Descent** first **all data** is passed forward through the network and all gradients are computed. The average of all gradients is then used to **update the weights once** after all data is passed through.

**Mini-Batch Gradient Descent** is the mixture of both taking some data inputs, passing them forward, computing their gradients, taking their average and updating the weights with that after each batch is passed through.


### 6. Adagrad
Adaptive Gradient Algorithm.

Idea:
  * individual per parameter learning rates

### 7. Adadelta
Only looks at past set number of gradients for update

### 8. RMSprop
Root Mean Square Porpagation

Idea:
  * individual, adaptive per parameter learning rates
  * rates are adapted based on the average of recent **first moment** of the gradients (first derivation)

RMSprop is actually nearly the same as Adadelta.

### 5. Adam
The name *Adam* is derived from *adaptive moment estimation*.

Idea:
  * individual, adaptive per parameter learning rates
  * rates are adapted based on the **first and second moment** derivative (momentum) of the gradients
    * an exponential moving average is calculated based on the gradient and the squared gradient

Parameters:
  * $\beta _1$ and $\beta _2$ control the decay of past learning rates for the moving average of those
  * $\alpha$ initial learning rate

Advantages:
  * easy to implement
  * computational efficient
  * little memrory requirements
  * invariant to diagonal rescaling of gradiants
  * well suited for problems with
    * many parameters or large data
    * noisy or sparse gradients
    * non-stationary objectives



Adam is supposed to combine advantages of AdaGrad and RMSprop



## Neural Networks as Autoencoders

## General Problems
* Overfitting/ Underfitting
* Generalization


# 05/06 - Backpropagation
BP is computational intensice during training. Ideas for improvement:
* parallel HW
* efficient implementation 
* faster gradient descent search (e.g. adaptive learning rates)
* selective choice of patterns
* efficient architecture

## Towards Efficiency
---

* skip training samples during back propagation
  * that are already well classified
  * they will probably not contribute much to the weight update because their contribution to the overall error ($\frac{\partial E} {\partial w_{i,j}}$) is small

* update weights only after each **epoch of training** not after each and not after all training data

### Dynamic adaptice learning rates

* Using momentum: 
  $\Delta w_{ij}(t) = - \epsilon * \frac{\partial E} {\partial w_{i,j}} + \alpha * \Delta w_{ij}(t)$
* **Rprop**: 
  * Use different learning rate $\epsilon _i$ per weight $w_{ij}$
  * Increase learning rate for a weight $w_{ij}$ if the weigt-gradient $\Delta w_{ij}$ show in the same direction as in the last time step: $\Delta w_{ij}^t > 0 \And \Delta w_{ij}^{t-1} > 0$

### Quickprop algorithm
* assumptions
  1. error/ weitght surface is a parabola
  2. the parabolas of each weight are independent (assumption not correct but it still works)

### Batch Normalisation

Problem: When updating the weights of layer l the input x distribution of layer l+1 changes.

Idea of BN: normalize each input x with the mean/expected input of the layer and the variance.

=> Higher learning rates and faster convergence

## Error and Loss Functions
---
* MSE
* Cross-Entropy Loss
* Mc Clelland error

### Mean Square Error

  Mean Square Error: MSE = $\sum_{j=1}^n (y_j- target_j)^2)$

  Problem with MSE: when there are many classes (n is big) the error of a single weight is small although it may be the only controbtution to the error. For example: Letter recognition with 1-hot-encoding: all are 0 except for one entry. If that entry is wrong, it is the only error source but still the MSE is small. <br>
  => Slow learning

  Solution: Use **Softmax** Function of the error

  Softmax:

  $\sigma (z)_j = \frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}}$

### Cross Entropy Loss

$L_{CE}(P, y) = - \sum_i y_i * \log(P_i)$

where
* P is the probability distribution of classes ($p(w_i | x)$) derived by the model
  * Usually calculated with the softmax funtion:
  * $p_i(out_i) = \frac{\exp(o_i)} {\sum_j \exp(o_j)}$
    * where $o_i$ is the ith entry of the output layer so representing the output for class i

### Mc Clelland Error

$E = - \sum_j ln(1 - (y_j - target_j)^2)$

## Activation Functions
---

  $y_j = f(x_j)$ for neuron j

where x is the sum of all inputs:

  $x_j = \sum_{i: <i,j>\in E} w_{ij} * y_i$ 

E = Set of all Edges/ Connections

* Step Function

  $f(x) = \begin{cases}
            0, &  x > 0 \\
            1, & x \leq 0
          \end{cases}$

  $\frac{\partial f}{\partial x} = 0$

* Sigmoid

  $f(x) = \frac{1}{1 + \exp{-x}}$

  $\frac{\partial f}{\partial x} = f(x) * (1 - f(x))$


* Softmax

  $f(x_j) = \frac{\exp(x_j)} {\sum_k \exp(x_k)}$

  $\frac{\partial f}{\partial x_j} = f(x_j) * (1 - f(x_j))$


* Hyperbolic tangent function (Tangens Hyperbolicus)

  $f(x) = tanh(x) = \frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}$

  $\frac{\partial f}{\partial x} = 1 - tanh(x)^2$


* linear function

  $f(x) = x$

* ReLU = Rectified Linear Unit

  $f(x) = \max(0, x)$

  $\frac{\partial f}{\partial x} = \begin{cases} 
                                      1, & x > 0 \\
                                      0, & x \leq 0
                                   \end{cases}$

* Softplus
  * Smoothed version of ReLU

  $f(x) = \log(1 + e^x)$

  $\frac{\partial f}{\partial x} = \frac{1}{1+e^{-x}}$

* Maxout

* ...

## Generalisation
---

Apparently the following equations are meant to approximate the test error given the training error, the number of parameters and the number of training samples.

for linear systems:

![gen_lin](figures/05_94.png)

for non-linear systems:

![gen_mul](figures/05_95.png)

### Optimize Generalisation
1. Destructive Methods

Reduce Complexity of model through regularization.
Meaning the network is punished for too large |w|
* weight decay
* weight elimination
* optimal brain damage => cwertain connections are removed from the network. e.g. 
  * those with small w_ij
  * or rather those with small effect on the overall loss (need to compute $\partial E / \partial w_{ij}^2$ => time consuming)
* optimal brain surgeon

Last summand = cost of a weight

$E = MSE + \lambda *\sum_{i,j} \frac{w_{ij}^2}{1 + w_{ij}^2}$

1. Constructive Methods

Iteratively incerease size of network
* cascade correlation
  * start with no hidden units
  * add hidden units/layers? iteratively
  * train the network for the newly added hidden layer
* meiosis network
  * adding weights is dependent on the uncertainty of the weights
* Automotive Structure Optimization

#### Dropout Training
A very popular method for generalisation is dropout training. Here during training some hidden units are dropped randomly for an epoch. Additionally the maximum weight update is limited.


# 07 - Unsupervised Learning

## Autoencoders
---
  * Linear Autoencoder
    * Iddea: Less hidden layer neurons than input or output neurons
    * Similar to linear compression method like PCA
    * Tries to find linear surface that most data can lie on
    * Not iuseful for cmoplex data
  * Non-linear autoencoders
    * Non-linearity through actication function

AEs can be trained with *SGD = Stochastic Gradient Descent*.

**Reasond for Autoencoders**
  * Conpression and reconstruction
  * MLP training assistance
  * feature learning
  * repressentation learning

An AE basically captues the data likelyhoood $P(X)$.

### **Data Compression and Reconstruction**
general ideas:
* reduce dimensionality of input pictures
* train network with *corrupted* input (e.g. noise, shift, effects, mask parts, ...) data train for regeneration skill

concrete applications:
* Generate high definition frames from small definition (SD?) frames in games
* denoise speech from microphones

### **Unsupervised Pretraining**

#### **1. Layer wise pretraining**

Case: a MLP with two hidden layers.

Procedure:
1. pretrain H1 as autoencoder hidden layer for the input layer. So $W_{in,H_1}$ is the weight matrix of the encoder
2. Pretrain H2 as autoencoder for H1
3. Train the network with the labels: called *"Finetuning"*

=> Each layer $H_n$ is pretrained as AE to reconstruct the input of that layer, namely $H_{n-1}$

Result
* lower classificatin error rate
* networks with 5 hidden layers converge

#### **2. Restricted Boltzmann Machines**
A BM is a  fully connected NN with some visible and some hidden units. It has no dedicated output layer and in general no concept of layers at all.
An RBM uses an **energy function** for the visible-hidden system

$E(V, H) = - \sum_{i=1}^m \sum_{j=1}^n w_{ij} * h_j * v_i - \sum_{i=1}^m v_i * a_i \sum_{j=1}^n h_j * b_j$

* $V :=$ Set of visible units
* $H :=$ Set of hidden units
* $m :=$ number of visible units
* $n :=$ number of hidden units
* $h_j :=$ output of hidden unit j. $h_j \in {0, 1}$
* $v_i :=$ output of visible unit i. $v_j \in {0, 1}$
* $b_j :=$ bias for unit j
* $b_i :=$ bias for unit i

To train a RMB the energy function is **minimized**.

The network spun by a RMB is also called **Deep Belive Network**.

### **Variational Auto Encoders**

Aim: Interpretation and Generation by forcing the hidden units to follow a Unit Gaussian Distribution.

We want the hidden layer to follow a known distribution so sampling gets easier...
To force the hidden layer to follow a distribution the loss function can be somehow adapted to include the parameters (e.g. $\mu$ and $\sigma$ of the normal distribution).

And that's all what's noted in the slides. Seems amazingly unimportant.

## Structure Prediction
---
* Idea: Given only part of the object predict the remaining => autocompletion with flexible NN architectures.
* These tasks leads to RNNs or CNNs/TDNNs or Transfer and Self-attention NNs.
* In the context of language a (neural) language model is mentioned. => How likely is a sentence/ how much sense does it make. Built on some input text/ language

Structure prediciton is and explicit and flexible method to deal with estimating the likelihood of data $P(X)$ that can be fatorized with a bias.


# 08 Hopfield Nets and Bolzmann Machines

## Hopfield Nets
---
### **Binary Hopfield Nets**

* single layer of fully connected units
* weights are symmetric and denoted as $T_{ij} = T_{ji}$
* each unit i has an activity value or state $u_i \in \{-1,1\}$ denoted as $\{-,+\}$
* network state = vecotr of unit states $U = (u_1, ... ,u_n)$
* Convergence:
  * convergence is necessary forthe HN to do something useful
  * symmetric HNs will converge to a stable state
  * asymmetric HNs can converge
* Input
  * $x_j = \sum_{i, i \neq j} T_{ij} * u_i$
* Binary actication
  * $u_j = g(x_j) = \begin{cases}
                      1, & x > 0 \\
                      0, & else
                    \end{cases}$
* Update procedure
  * asynchronous: one unit at a time
  * synchronous: all units in parallel
* Energy function
  * each state of the network can be described with a certain energy
  * $E = - 1/2 * \sum_j \sum_{i, i \neq j} u_i * u_j * T_{ij}$
  * Each change towards convergence leads to lower energy!!!


<img src="figures/08_05.png" style="height:250px;"/>

**Application: Associative Memory**
* A memory is represented by a stable (converged) state vector U.
* From an initial state the network converges to a stable state in which's attractor region it lies
* Can be use, e.g. for noise reduction of image completion.
  
**Problems:**
* Found stable state is not necessarily the most similar pattern
* spurious states can occur = stable states that do not represent memory
* not very efficient (100 neurons can store about 8 patterns)


## Boltzmann Machines
---
See also previous lecture on restricted Boltzman machines. 

* stochastic RNN
* problem: unconstrained connectivity => RBM

<img src="figures/08_37.png" style="height:200px;"/>

* binary states/ units:
  * visible
  * hidden
  * one bias
* Decision if state is active or not is **stochastic** and depends on the input. This is done to avoid local minima
  * $p(s_i = 1) = \frac{1}{1+e^{-z_i}}$
    * $s_i$ = output of state i
    * $z_i$ = input of state i = sum over weights times other states plus bias
* fully symmetrically conntected without self connections
* Energy function
  * same idea as in Hopfield nets
    * $E = - \sum_{i < j} s_i * s_j * w_{ij} - \sum_i b_i * s_i$
      * w = weight, b_i = bias for state i
  * E can be used to compute the probability of a certain input vector
    * $p(v) = \frac{\exp(-E(v)) } {\sum_u \exp(-E(u))}$

**Advantages of BM**
* With enough hidden units they can compute every function

**Problems of BM**
* slow training
* comnputational expensive

## Restricted Bolzmann Machines
---
Idea: No connection between input units and no connection between hidden units. => "Layer"

Usage: predict a user's rating for a film.

There is also more info on likelyhood of data and how to train an RBM.

Persistant Contrastive Divergence:
<img src="figures/08_67.png" style="height:300px;"/>


## Deep Belief Networks
---
* very deep networks
* hidden layers can be pretrained as autoencoders





# 09/10 - Speech Recognition with Time Delayed Neural Networks (TDNN)

Speech modeling approaches:
  * acoustic phonetic level
  * word level
  * sentence level

Challenges in Speech recognition
  * ambiguousity
  * compositionality
  * prone to "side effects" like emotino, age, gender, accent, dialect, ...
  * multilingual
  * latent content and meaning

  * Speech is not static
  * Shift (for example pitch, time delay)


Encoding of speech
* *"Formats"* := the resonance frequencies of the vocal tract transfer function
* time-frequency plots


## Time-Delay Neral Networks
---
  * Multilayer, nonlinear
  * \+ enable shift invariant learning!
    * hidden units learng features independent of precise location in time (or space)
  * for images this principle is adapted in Convolutional neural networks
  * weights among the different windows of one layer are shared.

<img src="figures/09_20.png" style="height:300px;"/>

Then there's a lot info on how the networks were build, e.g. that a "bdgptk"-classifier network was build from the hidden units of a "bdg" and a "ptk"-classifier network.

Some examples on where CNNs are applied today, Imagenet, AlphaGo, ...

## Word Models
---
* Problems
  * time alignment
  * endpoint detection
  * large vocabularies
  * compositionality of speech and language

* Approaches/ Solutions
  * NN-HMM-Hybrids
  * Multy State TDNN (MS-TDNN)
  * RNNs
  * End-to-End-Models

Some ideas to conquer time alignment:
* Linear Sequence Alignment
  * compute the distance of a word template to a reference template (=Sum of all Frame-to-Frame distances)
  * use distance to derive an alignment between the word and the reference
  * \+ can handle different speaking rates
  * \- cannot handle varying speaking rates during same utterance => Non-Linear Alignment needed
* Time Warping = Alignment through time

<img src="figures/10_11.png" style="height:400px;"/>


## Speech Recognition
---
Componentes of a speech recognition system:

<img src="figures/09_64.png" style="height:300px;"/>

* Recognizer components are all NNs

Goal of Speech Recognition:
  * Given acoustic data A
  * fined word sequence W
  * such that $P(W | A) = \frac{P(A | W) * P(Ww)}{P(A)}$ is maximized

### Hidden Markov Models
A very vague reminder on HMMs:

* structure: <img src="figures/09_70.png" style="height:300px;"/>
* Foward Algorithm: Sum
* Viterbi Algorithm: Max
  * Viterbi maximized the probability of a state sequence Q
* Both find some state seqzence Q

### Hybrid Models

**#### NN-HMM Hybrids**

  * NN for classification of phonemes
    * aA NN computes the posterior class probability $P(w_j | data)$
  * HMM for alignment and integration into words

<img src="figures/09_70.png" style="height:300px;"/>

**#### MD-TDNN**

<img src="figures/10_39.png" style="height:300px;"/>
 
 ### Recurrent Neural Networks
 cf. section 16

 ## Language Models
A language model is a probability distribution $p(w_i|w_1, ..., w_{i-1})$ for a word/ character w_i given its history as n-gram.

 * some info on n-grams and sequence probabilities
 * Quaity Measure: LogProb: $H(W) = - \frac{1}{n} \sum_{i=1}^n \log_2 Q(w_i| \psi(w_1, ... w_{i-1}))$



# 11 Speaker Independence

General approaches:
  * Build one speaker independent model
    * hard to separate
  * Build many speaker dependent models
    * not enough data => bad generalizatino

## Methods to achieve speaker independence
  * Model invariance towards
    * frequency shift
    * tilt
    * compression, ...
    * 
  * Adaption
    * How fast does ne listener realize a speaker change/ what speaker is speaking?
    * humans fast: in one or two syllables
    * 
  * Normalization
    * correcting environment variables
    * mapping a speaker to a *Standard Speaker*, normalizing speakers

  * i-vectors
    * i-vecotrs are Eigenvecotr derived informations on a speaker
    * i-vectors can be extracted from a spoken sequence and be used to characterize a speaker.

### SAT - Speaker Adaptive Training
1. Train DNN as usual
2. Train an i-vector NN that normalized the i-vector of a speaker. Keep the DNN parameters fixed.
3.  Train the DNN parameters in the new feature space created by the i-vector NN. Keep i-vector NN parametes fixed.

<img src="figures/11_23.png" style="height:300px;"/>


### Multi-Speaker Reference Model
Idea: Have multiple language/ speech recognition models, each trained on a different speaker. Use a general model that first classifies the speaker and then uses the model.

<img src="figures/11_25.png" style="height:300px;"/>

Further: Meta-Pi Net
Don't try to distinguish the speaker, but try to optimize the performance of the overall error/ loss of the single speaker dependent models

<img src="figures/11_26.png" style="height:300px;"/>

### Multi-Lingual Model
**Cross Language DNNs with Language Universal Features**
Idea: Borrow knowledge from one language to train another language model.

* Language code:
  * LID Language identity
    * 1-hot vector encoding of identity
  * LFV Language Feature Vector
    * Encoding of a language's properties
    * how it sounds, how it is articulated

<img src="figures/11_33.png" style="height:300px;"/>

=> A network that can dynamically switch between different language models


# 12 - Handwriting Recognition

**Characterisation**

* offline recognition
  * information
    * coordinates x,y
    * tilt
* online recognition
  * additional information:
    * coordinates x,y fand time t
    * pressure
    * pen_down and pen_up events
    * velocity

<img src="figures/12_55.png" style="height:300px;"/>

## On-line recognition

### Approaches
* Elastic matching
* Break down into stroke levels
  * '|' + '-' + '|' = H
  * problem: complex rules
* explicit segmantation
* elastic matching
* input and output segmenation
  * often borders are hard to distinguish
  * allows to introduce some a priori knowledge, e.g. about statistical distribution of letters

### Normalization
Remove undesired variability in online handwriting
  1. Baseline normalization (deskewing)
     * baseline ist detected using Expectation Maximization  
     * rotate till baseline is horizontal 
  2. bezier normalization (smoothing)
     * bezier algorithm approximatex missing data points
     * moving average used to delete noise and connect points smoothly
  3. Skew normalization
  4. size normalization
     * sescaling 
  6. resampling from temporal to spatial equidistance
     * resample points to make them have equal distance to each other 
  7. removing delayed strokes
     * like i dots or t bars that occur temporally delayed 
   
<img src="figures/12_72.png" style="height:300px;"/>

### Word Modeling
Again something on Markov Chains, Viterbi Algorithm and MS-TDNNs.
Also Flat Search is mentioned.

And a Tree Seach Algorithm.

## Off-line recognition
* Sign translatino
* like object detection + image recognition + language recognition (+ translation)




# 13 - Computer Vision

## The Computer Vision Task
* Discriminative Task
* Generative Task
  * Image Transfer
  * Enhancement
  * Generation



## Object Recognition
* Classification
* Localization
* Detection
* Segmentation

### Classifiation
Typical framework:

<img src="figures/13_06.png" style="height:140px;"/>

* Feature examples
  * SIFT - shift invariant from transformation
  * HoG - Histogram of gradients
  * RIFT - radiation invariant from transformation

### Shift invariance
Def: "The outputs are independent to the shift along one or several dimensions of the feature space."

<img src="figures/13_18.png" style="height:300px;"/>

### CNN - Convolutional Neural Network
<img src="figures/13_20.png" style="height:250px;"/>

Consists of
  * Convolutional Layers
    * using filters
    * can have activation functions
      * often ReLU or tanh
    * one convolutional layer shares a weight matrix throughout its filters
  * subsampling layers
    * aggregate convoluted features to reduce the number of parameters
    * does not learn any parameters
    * aggregated regions often do not overlap
    * popular aggregations
      * mean
      * max
      * probabilistic pooling
  * fully connected layers towards output
    * typical MLP procedure with 
      * input $x$
      * weight matrix $W$ 
      * bias $b$
      * and actication function $f(W*x + b)$

**CNN for take away:**
* A CNN in contrast to a MLP can deal with shift-variance of features in time and space.
* They also reduce the number of parameters through convolution.
* next layer can be resized choosing convolution or pooling parameters
  * kernel size
  * number of kernels
  * padding
  * stride
  * convolution:

  $height = \Bigg\lfloor \frac{h_{prev} - f + 2 * p}{s} \Bigg\rfloor + 1$

  $width = \Bigg\lfloor \frac{w_{prev} - f + 2 * p}{s} \Bigg\rfloor + 1$
  * pooling

  $height = \Bigg\lfloor \frac{h_{prev} - f}{s} \Bigg\rfloor + 1$
  
  $width = \Bigg\lfloor \frac{w_{prev} - f}{s} \Bigg\rfloor + 1$


### Learning hierarchical features
* Learned features can be seen as building blocks of an image
* High level = more conntent
* Low level = more detailed information, concrete local behaviour/ pixels
* Use transfer learning on those builing blocks

#### Transfer learning
1. Use pre-trained network
2. remove output layers
3. fix pretrained layers and only train new layers and added new output layers

#### Neural Style Transfer
The idea of hierarchical features can also be used for style transfer. E.g. transfer the features learned in a low-level layer to another input:

* Naive idea: optimize loss

  $Loss L = \alpha * L_{content} + \beta * L_{style}$

  * Problem: optimization is slow
* NN idea: train a separate NN to generate a style transfer image

<img src="figures/13_67.png" style="height:250px;"/>


## Single Shot Learning
Problem: Face Recognition. New face is new class with only one data sample

Approaches:
* Siamese Network
  * Run same network with two different inputs
    * compare output
    * if distance small, same class


## Object Detection

<img src="figures/13_76.png" style="height:250px;"/>

* For example sliding window
* Recursive CNN

## Semantic Segmentation
* Yolo
* Also Sliding windows










# 14 Parallelism, Hardware Acceleration and Frameworks
* Parallelism:
  * User vector oerations instead of for loops (SIMD operations)
  * compute the results of each row in parallel (Libraries such as BLAS)
  * Use GPUs for the parallel stuff
* Frameworks abstract from GPU usage
  * Static computational graph
    * graph is created ahead of time and then executed for some data
  * dynamic graph builders
    * models are written like regular programs
  * Automatic differentiation
    * symbolic differentiation (= analytical) has long expressions and a lot of repetitions
    * numeric differentiation is slow and has rounding errors
    * idea:
      * break down programm into a sequence of primitives with known derivative
      * during forward pass store the intermediate results
      * during backward pass use the known rules on how to compute the gradients to simply concatenate the gradient mechanically.

<img src="figures/14_08.png" style="height:150px;"/>



# 15 - Audio-Visual Speech (Lipreading) and Focus of Attention

  * McGurk effect: audio and visual informatino are fused

Motivation
  * use lipreading to improve acoustic speech recognition

Result
  * lip reading supported acoustic speech recognition is better with more environmental noise

## Tracking
  * OF faces
    * using color model
      * \+ fast, orientation invariant, stable representatin
      * \- device dependant user dependant envornment dependent
    * using fix points like eyes, nose, mouth corners
  * Of lips
    * search for eye pupils
    * (localize nostrils)
    * predict search region for lip corners

## Fusion of Information

* word level
* phoneme level
* feature level

## Tracking the Focus of attention
* in meetings
* in classrooms

Main questions
* where is someone looking?
* what is he/she looking at?
* is that person awake?


# 16 - Recurrent Neural Networks

* Used for generative language models with a conditional probability distribution for each word or character

  $P(w_n | w_1 w_2 ... w_{n-1})$

## Elman/ Jordan Networks
RNNs with limited horizon. The hidden state (elman) or the output state (jordan) of timestep t is transferred only to the next timestep t+1.

### Elman Network

<img src="figures/16_14.png" style="height:300px;"/>
<!-- ![Overview](figures/16_14.PNG) -->


### Jordan Network
<img src="figures/16_16.png" style="height:300px;"/>
<!-- ![Overview](figures/16_16.PNG) -->

## Simple Recurrent Network
In and RNN everything is handled in sequences. The input is a sequence $X$ of (data-) vecotrs, the output a sequence of (probably) vecotrs $O$ and the labels a sequence of groundthruth-vectors $Y$ - for example 1-hot-encoded in case of classification.

An RNN "rolled out" in time:

<!-- <img src="figures/16_19.png" style="height:300px;"/> -->
![Overview](figures/16_19.PNG)

Where the hidden state of timestep t is give by

$H_t = f_{act}(W^H * H_{t-1} + W^X * X_t + b)$

with
%f_{act}$ bbeeing an activation function, $W$ being the weight matrix and X being the input of timestep t.

### Variations of RNNs

* One-to-One
  * = MLP
* Many-to-One
  * Often used for sequence-level classification
  * BPTT

  <!-- <img src="figures/16_25.png" style="height:250px;"/> -->
  ![Overview](figures/16_25.PNG)


* Many-to-Many
  * Composite Loss function L = L_1 + L_2 + ...

  <!-- <img src="figures/16_31.png" style="height:250px;"/> -->
  ![Overview](figures/16_31.PNG)

* One-to-Many
  * Rare version
  * for example for music generation

  <!-- <img src="figures/16_34.png" style="height:250px;"/> -->
  ![Overview](figures/16_34.PNG)


* Sequence-to-Sequence
  * See section 17

## Back Propagation Through Time (BPTT)

* Forward Pass
  * Maintain Memory of hidden layers
* Backward Pass
  * Back propagate the error through the hidden layers
  * Gradients over shared weights are summed up

Problem: 
  * Vanishing and Exploding Gradients!!!
    * Due to long dependencies through time
    * If the max Eigenvalue of $W^H > 1$ => $\frac{\partial L}{\partial H_T}$ is likely to explode!
    * If < 1 => vanish

Simple Solutions for exploding Gradients
* Gradient Clipping:

  With  $g = \frac{\partial  L}{\partial w}$ do
    * if $||g|| > \mu : g = \mu * \frac{g}{||g||}$

## Long Short-Term Memory Networks (LSTM-Networks)

* The RNN has additional memory cells.
* Into these cells inputs can be 'committed'/'stored'.
* Later inputs 'erase' stored content

Sketch:

  <!-- <img src="figures/16_46.png" style="height:250px;"/> -->
  ![Overview](figures/16_46.PNG)

Operations on memory cells:

1. Forget

   * The forget gate is a neuron F which is connected 
     * to the input and the hidden layer as inputs with weights $W^{FX}$ and $W^{FH}$
     * and the or a memory $C_j$ at the output side.
     * Its output id $F_{tj}$
   * Remove information from cell $C_{j}$ in timestep $t$
   * Removing depends on current input $X_t$ and previous memory $H_{t-1}$

     $F_{tj} = sigmoid(W^{FX} * X_t + W^{FH} * H_{t-1} + b^F)$

   * Memory update: $F_{tj}$ close to 0 => delete/ forget

     $C_t = C_{t-1} * F_{tj}$


2. Write or Add

   * Add new information to memory $C_{j}$ in timestep $t$.
   * Input gate $I_{tj}$ the same as forget gate

     $I_{tj} = sigmoid(W^{IX} * X_t + W^{IH} * H_{t-1} + b^I)$

   * The content to write to the memory is 

     $\tilde{C}_t = tanh(W^{CX} * X_t + W^{CH} * H_{t-1} + b^I)$

   * The update is done with

     $C_{tj} = C_{t-1,j} + I_{tj} * \tilde{C}_{tj}$

3. Output (read)

  * Read from cell $C_j$ at timestep t and store it in hidden state H
  * Output gate

    $O_{tj} = sigmoid(W^{OX} * X_t + W^{OH} * H_{t-1} + b^O)$

  * Hidden State update

    $H_{t} = O_{tj} * tanh(C_{tj})$


## Side note on feature map and feature embedding

* **Feature mapping** means, e.g. to display and use a word encoded as a one-hot-vector
  * $V = \{w_1, w_2, ..., w_n\}$ 
  * with $w_i$ beeing the one-hot-vecotrs of each word/ character in V. $w_i = [0,0,...,1,0,0,...]$
* **Feature embedding** or **word embedding** then means a **linear transformation $E$** of the high dimensional, sparse vector $w_i$ to a more dense vector $e_i$. 
  * $e_i = W_E * w_i$



# 17 - Sequence-to-Sequence Model

Standard loss for RNNs: Cross-Entropy Loss

$L_{Ce} = - \sum_i y_i * \log(p_i)$

Difference between Cross-Entropy Loss and Mean Square Error
  * CE assumes an underlying **Multinomial** distribution
  * MSE assumes an underlying **Gaussian** dristribution
    * $MSE = \sum_i (y_i - p_i)^2$

## Application of LSTMs
Principle: Concatenate two RNNs via the last hidden layer of the first.
Usage:
  * Sentence Completion
  * Translation!

  <img src="figures/17_32.png" style="height:350px;"/>

Works well for short sentences, but bad for longer > 10 words.

Funny trick: reversing the source sentence increases performance

* Encoder Decoder with LSTMs
  * Work also for images
  * problems
    * representation efficiency
    * gradient starvation

  <img src="figures/17_51.png" style="height:350px;"/>

## Sequence-to-Sequence with attention
Goal: find the alignment between encoder and decoder.

Typical RNN Question: "Which word is the most likely one to appear after this sequence."
Encoder-Decoder Question: "Which wird is responsible to generate the first word"

Idea use additional attention node C that sees all hidden states of the encoder

  <img src="figures/17_60.png" style="height:350px;"/>

where

* $C_0 = \sum_i \alpha _i^0 * H_i^e$
* $\alpha _i^0$ is the relevance of state $H_i^e$ for the attention state $0$
  * $\alpha _i^0 = W_2 * \tanh(W_1 * [H_0, H_i^e] + b_1)$
  * normieren auf $\sum_i \alpha _i = 1$: softmax
    * $\alpha \larr softmax(\alpha)$
* $\hat H_t$ is the combination of $C_t$ and $H_t$ and can be realized as e.g.
  * $\hat H_t = \tanh(W * [C_t, H_t])$
  * a RNN

Advantages of Attention
  * no gradient starving
  * attention weights allow heat map to check if alignment is correct

Other applications
  * Machine Translation
  * Speech Recognition
    * Listen, attend, spell
  * Image description generatin
    * also, point out focus of an image description
  * Music Automation transcription
  * automatic summarization

S2S is not good at
  * Conversational bots


# 18 - Transformer model - Attention is all you need

Inventions that accelerated Deep Learning

## Dropout
* Mask some neurons during training.
  * All connections to/ from these neurons are ignored
  * their weights are not updated
  * they are still used during testing
  * some scaling of the masked input-layer must be applied to shift the expected size between training (smaller) and testing (bigger)
    * e.g. scale training with $\frac{1}{1-p}$
    * or scale testing with $p$

## Batch Normalization

* Problem: During training updating a lower layer changes the input distribution of the next layer
* Idea: Mean and variance normalization step between layers
* Advantages: 
  * faster training
  * gradients less dependent on scale of parameters
  * allow higher learning rates
  * combat saturation problem

Normalizatio:

$\hat X = \frac{X - \mu}{\sqrt{\sigma^2 + \epsilon}}$

Scale and Shift

$X^N = \gamma * \hat X + \beta$

### Other Normalizations
as alternatice to (Mini-)Batch Normalization

* Layer Normalization
* Instance Normalization
* Group Normalization

<img src="figures/18_19.png" style="height:200px;"/>


## Residual Learning
Some connections added to the network which skip single layers...

<img src="figures/18_23.png" style="height:250px;"/>



## Attention Mechanism

* Encoder Decoder can be seen as a Neural Turing Machine
* The weihgts $\slpha$ fopr generating the attention states C are generated with a Neural neutwork

Usage/ Idea of attention
* detect correlation between 
  * source and target words
  * natural language and image features
  * questions and memory entries
* Use learnt coefficients $\alpha$ to detect these
* There is always some query and some key-value pairs as data



### Obsessions with Modeling
1. Computation efficiency
    * Use Deep TDNN with kernels that see two words => Residual Connections 

<img src="figures/18_44.png" style="height:200px;"/>

Desiderata = what is missing but wished for
* use parallel operations
* preserve resolution
* forward and backward path should be short
* runtime should be linear

## The Transformer Model
 Sequence Modelling with Attention

Use self-attention to model patterns in sequences. Self-attention means that the own input $X$ is used as query $Q$ as well as as key $K$ and value $V$ pair. $Q$, $K$ and $V$ are all some sort of additional hidden states. Hoewever the hidden state $H$ is derived from the $\alpha$s and the data-values $V$.

<img src="figures/18_54.png" style="height:200px; float: right;"/>

  $Q_t = W^Q * X_t + b^Q$  <br>
  (red)

  $K_t = W^K * X_t + b^K$ <br>
  (grey)

  $V_t = W^V * X_t + b^V$ <br>
  (green)

Matching score $e_{ij}$:

  $e_{ij} = Q_i * K_j$

<img src="figures/18_55.png" style="height:150px; float: right;"/>

  $\alpha _{ij} = sfotmax(e_{ij}) = \frac{\exp(e_{ij})}{\sum_k^N \exp(e_{ij})}$

Attention hidden state:

  $H_{att, t} = \sum_k^N \alpha _{tk} * V_k$
  <br>

<img src="figures/18_56.png" style="height:150px;"/>

<img src="figures/18_57.png" style="height:400px; float: right;"/>

More hidden layers can be added on top of $H_{att, t}$:

$H_R = X + H_{Att}$

$H_{drop} = Dropout(H)_r$

$H_{norm} = LayerNorm(H_drop)$

<br><br><br><br><br><br><br><br><br><br><br><br>

A sequence model with attention

<img src="figures/18_62.png" style="height:400px;"/>


Problems with attention based sequence models
* does no take into account word positions

Workaround:
* add temporal/ positional information
  * by adding sine/ cosine

Some info
* BERT = Pretrained self-attention models for unsupervised learning
* Transformer is 
  * computationally fast (compared to RNN or CNN)
  * very memory expensive






# 19 - Natural Language Processing (NLP)

NLP Tasks

<img src="figures/19_03.png" style="height:400px;"/>

* Named Entity (NE) recognition
* So NEs can be replaced with a single token

## A dialog system

<img src="figures/19_10.png" style="height:300px;"/>

* ASR
  * NN
* Text proprocessing for NLU
  * NE recognition
  * byte pair encoding
  * word embedding => reduce dimensionality and make space more dense
* DM
  * e.g. ask if an object is ambiguous
* NLG
  * NN

## End-to-End Dialog System
* error propagates better thorugh the system
* no intermediate dataset needed
* does not throw away information like the pitch of voice

<img src="figures/19_23.png" style="height:150px;"/>

Models for End-to-End Dialog learning
* Encoder Decoder with Attention
* Hierarchical Recurrebnt Encoder-Decoder
* Transformer Model
* Memory Networks
* Dynamic Networks

### Memory Networks

<img src="figures/19_32.png" style="height:400px;"/>
<img src="figures/19_33.png" style="height:400px;"/>

### Dynamic memory networks

<img src="figures/19_36.png" style="height:400px;"/>

* GRU
* Beam Search


# 20/21 - Reinforcement Learning
= Learn to use a good sequence of actions


---
Agenda:
## RL
* Markoc Decision Process (MDP) and Optimality
* Ingredients of an RL Agent
* Algorithm Taxonomy

## Optimal Control: Solving MDP
* Value Iteration
* Policy Iteration

## Value-based Methods
* TD-Learning
* Q-Learning
---

## RL

* Can be seen as a Markov Decision Process

<img src="figures/20_08.png" style="height:200px;"/>

Components
* Q-Function
* V-Function
* Transition function p(s' | s, a)
* Reward function r(s, a)

<img src="figures/20_18.png" style="height:400px;"/>


## Optimal Control: Solving MDP

## Value-based Methods


# 22 - Generative Adversarial Networks (GANs)

# 23 - Adversarial Examples

# 24 - Neural Networks for Control
Situation: The thing to be controlled is the "plant". The input of the plant is $u$ the output is $y$ and the desired output is $y^*$. E.g. for robot arm $u$ would be the joint angles, $y$ would be the TCP (Tool Center Point) position.

General idea: use a NN to learn a controller. A controller is the part which defines what input $u$ a plant must get to behave in a desired way $y$.

Two models useful for this task:
* a forward model: given $u$ calculate the predicted output $\hat y$
* a reverse model: given some output $y$ calculate the given input $u$. (This is actually what the controller does)

## Training a controller
... is difficult because there may be more than one legit solution (like ambiguity of joint angles).
So training with normal Gradient descent, which would try to learn the average solution, does not work, since the average solution actually is no solution. Example: Two joint angles lead to the same TCp position. However, their average leads to a totally different position.

So other training techniques are required like...

**Distal Learning Approach**
* First train the forward model with normal BP
* Freeze forward model
* Use Forward model to propagate error through it to the controller or inverse model
* Train controller 

Distal learinng is *goal directed* meaning it is only good in the region you care about since it does not sample the whole space... whichever that is.

Also the forward model is apparently not precise, however, that is quite *good enough*.

<img src="figures/24_18.png" style="height:150px;"/>

**Static and Dynamic**

In this case static means that there is no influence of the state of the plant on the outcome of an input. Dynamic then again means that a different state $s_t$ leads to a different outcom $y_t$ for the same input $x_t$ as in another state.

This can be incorporated in a *Distal Teacher with State* model. And the training actually is some sort of RNN training through time.

The teacher idea is:
* environment teaches forward model
* forward model teaches controller/ inverse model

<img src="figures/24_23.png" style="height:150px;"/>

Lastly some application examples are given
* Truck Backer-Upper
* Arm dynamics


# 25 - Summary Lecture

* Perceptron
  * linear separator $g(x) = \sum_{i=0}^n w_i * x_i$; $x_0 = 1$
  * $y_i \in  \{-1, 1\}$
  * $y_i * g(x_i) \leq 0$ => wrong class
  * update:
    * for all wrong classes
      * $w_i = w_i + y_i * x_i$
      * so label -1 => w = w - x

* MLP
  * input layer $X$
  * nonlinear activation function
    * ReLU
    * sigmoid
    * tanh
    * softmax
  * hidden layer $H$
    * $h = f(W*X + b)$
    * $h_j = f(\sum_i w_{ij} * x_i + b_j)$
  * output layer $O$
    * normalize output e.g. with softmax for porbabilies
  * Loss functions
    * Cross-entropy: multinomial distribution of O and Y
    * Mean-square-error: gaussion distribution ...
    * Binary Cross-entropy: prediction is independent binary unit (e.g. pixels)

* Stochastic Gradient Descent
  * is analgroithm how to update weights
  * learning rate $\alpha$
  * update $w_{t+1} = w_t - \alpha * \frac{\partial L}
    * minus because of "descend"{\partial w_t}$
  * Other variants like Batch and Mini-Batch SGD

* Backpropagation
  * is an algorithm to efficiently compute gradients that uses
    * chain rule
    * dynamic programming
  * forward pass and backward pass
    * forward pass needs to store the activations of hidden states
  * there are others like numerical gradient calculation

* TDNN
  * actually does some convolution on time data

* CNN
  * convolutional layer
    * filters/ kernels and feature maps
      * feature map = output of convolutional filter 
    * $k$ = number of kernels/ filters
    * one kernel/ filter applies to all channels if size is $f * f * c$; $c$ = number of channels
      * it then creates one feature map from all channels
      * probably by adding the resulted feature maps
    * \# of parameters per conv. layer = $f*f*c*k$
  * pooling layer

* RNN
  * one input at a time
  * hidden state stores info on past input
  * Elman and LSTM
  * weight sharing over time

* BPTT
  * Many-to-One
  * One-to-Many
  * Many-to-Many
  * gradient vanishing and exploding
    * vanishing: LSTM
    * exploding: clipping, normalization, weight decay

* S2S
  * use two RNNs
    * one to encode the input in a hidden state
    * one for the task like translation (decode)
  * modular constructed
    * can combine CNNs and RNNs
  * suffer from compression of the encoder

* Attention and the Transformer Model
  * attention means that the hidden states of the encoder are accessible from each state in the decoder  via an attention state
  * Transformer is a realiisation of the attention idea
  * \# of operations needed to relate two arbitrary points in the sequence = 
    * LSTM: $O(n)$
    * CNNs/TDNNs: $O(log_k n)$
    * Attention: $O(1)$
  * computational complexity (if all matrix multplications have constant cost)
    * LSTM: $O(n)$
    * CNNs/TDNNs: $O(1)$ (because it basically is one matrix multiplication)
    * Attention: $O(1)$


* Unsupervided Learning and Auto Encoder
  * Auto-Encoder and Variational AE
  * Unsupervised Learning is somewhat interpreted as superwised learning to reconstruct the input 
  * VAE introduces stochastic elements in BP


* Generalization
  * Underfitting
    * Model too general (wrong architecture)/ not enough capacity/ not trained enough
    * => better architecture with higher capacity
  * Overfitting
    * Model too complex/ trained too long on training data
    * many solutions, mostly punish and restrict the network
  * Weight Decay
    * minimize the magnitude of the weights
    * $L_{wd} = \beta * \sum_i w_i^2$
    * make network sparser
  * Dropout
    * hidden weights are masked out randomly during training
    * always makes the network train longer with higher training error
    * needs some scaling when testing path
    * introduces restrictions
  * Batch Normalization
    * Noise is eliminated


* Adaptive Learning Techniques
  * SGD with Momentum
  * Adagrad
  * RMSProp
  * Adam
    * the more momentums are used and stored, the less space efficient the algorithms are


Low Priority Section
* Neural applications
  * Speech applications
  * Hand-writing recognition
  * NLP
* GANs and adversarial attacks


Gneral comments on exam tasks
* CNN Question
  * The combination (3, 1, 1) means f=3, stride= 1, padding = 1
    * This combination for a filter preserves the input resolution