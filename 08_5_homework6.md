### Week 8 homework (HW #6)

Due Fri 18 Mar

The aim of this homework is to consolidate understanding of neural networks through coding.


#### 1. Install Keras

It's easiest to install `keras` by carefully following the directions from https://hastie.su.domains/ISLR2/keras-instructions.html, which is the ISLR2 book website. These instructions were working recently for Windows, Mac and Linux. Installation should be relatively painless and not disruptive to your current setup (e.g. projects relying on Python or Tensorflow) but check the preamble in the instructions if you are concerned. The installation will set up an instance of Miniconda that will isolate Python and Tensorflow from other versions on your computer. On Windows 10, the default installation is into /Users/<username>/Documents/islr-miniconda and a new start menu item is added called Miniforge3.

#### 2.  Code a network by hand

Extend the "by-hand" code in [07_2_ants_neural_net.R](07_2_ants_neural_net.R) to a 2-layer neural network, each with 5 nodes. Plot the model predictions. Use these weights and biases (n.b. you don't need to train the model):

```R
w1 <- c(-1.05834079, -0.8540244,  0.8608801, -0.53949207,
        -0.64411271,  0.5407082,  1.2520101, -0.71171111,
         1.16630900, -0.2184951, -0.1495921,  0.18796812,
         0.08298533, -0.1178127,  0.8332534,  0.30929375,
        -0.41105017,  0.3603910,  1.1532239,  0.05233159) %>%
    matrix(nrow=5, ncol=4, byrow=TRUE)

b1 <- c(0.1909237, 0.5486836, -0.1032256, 0.6253318, 0.2843419)

w2 <- c( 0.04039513, -0.147718325, -0.3422508, -0.79168826,  0.4339046,
         0.79774398,  0.368297666,  1.2922773,  0.54198354,  1.0874641,
         0.60440171,  0.959372222,  0.1165112, -0.05803596,  0.5460970,
        -0.18009312,  0.344686002,  0.5326685, -1.21680593,  0.2390731,
        -0.21000199,  0.008643006, -0.5922273,  0.16980886, -0.5996938) %>%
    matrix(nrow=5, ncol=5, byrow=TRUE)

b2 <- c(-0.29183790, 0.32845289, 0.32393071, 0.06806916, -0.01153159)

w3 <- c(-0.3925169, 0.8072395, 1.398517, -0.7064973, -0.3754095) %>%
    matrix(nrow=1, ncol=5, byrow=TRUE)

b3 <- 0.3231535
```

Compare this 2-layer model to the single-layer model. Describe qualitatively (i.e. make a comment) how the predictions differ.

Hint: don't start from scratch, just add a few lines of code here and there where needed. The goal of this is to gain greater understanding of the algorithm.

**Push your code to GitHub**

#### 3. Neural network in practical use

Train and tune a neural network to compare to the boosting algorithm you used in the previous homework for plant species "nz05".

Suggestions:

* Use a feedforward network
* Use binary_crossentropy loss
* Use a batch size of 32 (we could tune this but use the default for now)
* Tuning: we don't have the computation resources to try many combinations or to do k-fold CV. Here is a strategy:
  * use the cross validation option built into keras (i.e. `fit()` argument `validation_split=0.2`)
  * try four different architectures, e.g. 25 wide, 50 wide, 5x5 deep, 5x10 deep
  * try adding dropout regularization to the layers with 0.3 as a default rate

* Early stopping: often it is advantageous to stop learning after some number of epochs to prevent overfitting (i.e. when you see the validation error start to go back up)
* Compare to the boosting model. Which model gave the best predictive performance?
  * We can't formally compare the models here. We would need to first set aside a test set to make the comparison, i.e. we'd need a three-way split: train-validate-test, only comparing the models using the test set after tuning the models on the validation set.
  * Nevertheless, does the neural network get within the ballpark of the boosting model comparing the k-fold CV mean accuracy of the boosting model to the CV accuracy of the neural network?

* Plot predictions as you did for the boosting model

**Push your code to GitHub**

