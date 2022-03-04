#' ---
#' title: "Convolutional neural network"
#' author: Brett Melbourne
#' date: 01 Mar 2022
#' output:
#'     github_document
#' ---

#' We are using a standard benchmark dataset, CIFAR100 but subsetted to images
#' in ecological categories. This script has minimal commentary.

#+ results=FALSE, message=FALSE, warning=FALSE
library(ggplot2)
library(dplyr)
library(keras)
tensorflow::set_random_seed(2726)

#' Read local copy of the data labels

label_names <- read.csv("data/cifar100_fine_label_names.csv")

#' Download the CIFAR100 dataset. Warning: 169 MB. Since it's large, we'll store
#' it locally in a directory for temporary data and remember to put it in
#' `.gitignore` so it doesn't get uploaded to GitHub. It will take a minute or
#' two to download.

if ( !file.exists("data_large/cifar100.RData") ) {
    cifar100 <- dataset_cifar100()
    save(cifar100, file="data_large/cifar100.RData")
} else {
    load("data_large/cifar100.RData")
}

#' We see it's a list with a `train` and `test` object

str(cifar100)

#' Subset to the ecology images (including human)

ecosubset <- subset(label_names, ecology==TRUE)
head(ecosubset, 20)
train_eco <- which(cifar100$train$y %in% ecosubset$label)    
test_eco <- which(cifar100$test$y %in% ecosubset$label)
x_train <- cifar100$train$x[train_eco,,,]
y_train <- cifar100$train$y[train_eco,, drop=FALSE]
x_test <- cifar100$test$x[test_eco,,,]
y_test <- cifar100$test$y[test_eco,, drop=FALSE]

#' What do we have?
#' 

#' For x we have 30500 images, each 32 x 32 pixels in 3 channels (RGB), arranged
#' in a 4D array. Pixel values range from 0-255.

dim(x_train)
class(x_train)
range(x_train)
hist(sample(x_train, 5000))

#' For y we have integers coding for 61 categories arranged in a 2D array (1
#' column matrix).

dim(y_train)
class(y_train)
head(y_train)
sort(unique(y_train)) #61 ecological categories

#' Data preparation 1: convert image data to 0-1 scale.

x_train <- x_train / 255
x_test <- x_test / 255

#' Data preparation 2: generate new integers for the 61 ecology categories. This
#' must have integer category labels that range from 0 to m - 1, where m is the
#' number of categories. This is because tensorflow and python array indices
#' start at 0 (compared to R, where indices start at 1). 

ecosubset$ecolabel <- 0:60

#' Here are the corresponding original and new labels (`label` vs `ecolabel`).

head(ecosubset, 10)

#' Now make the new integer response using the lookup table in `ecosubset`.

for ( i in 1:nrow(y_train) ) {
    y_train[i,] <- ecosubset$ecolabel[ecosubset$label==y_train[i,]]
}
for ( i in 1:nrow(y_test) ) {
    y_test[i,] <- ecosubset$ecolabel[ecosubset$label==y_test[i,]]
}

#' Check the first 10 (e.g. compare with ecosubset above)

data.frame(y_train[1:10,], name=ecosubset$name[y_train[1:10,]+1])

#' Data preparation 3: convert integer response to a dummy variable matrix
#' suitable for keras/tensorflow. We'll use the `to_categorical()` function from
#' `keras` to do that.

y_train_int <- y_train #keep a copy of the integer version
y_train <- to_categorical(y_train, 61)

#' The result is a matrix with 61 columns, 1 column for each category of
#' organism.

class(y_train)
dim(y_train)

#' Looking at a portion of the matrix (upper left 6x14) we see we have rows of
#' zeros and ones, with a 1 in the column that represents the category of the
#' organism in the image.

y_train[1:6,1:14] 

#' There are 500 images in each category

colSums(y_train)

#' Random selection of images

par(mar=c(0,0,0,0), mfrow=c(5,5))
for (i in sample(1:dim(x_train)[1], 25) ) {
    plot(as.raster(x_train[i,,,]))
    text(0, 30, labels=ecosubset$name[y_train_int[i,]+1], col="red", pos=4)
} 

#' Each image has 3 channels: RGB

par(mar=c(0,0,0,0), mfrow=c(2,2))
plot(as.raster(x_train[200,,,]))
text(0, 30, "color", col="white", pos=4)
plot(as.raster(x_train[200,,,1]))
text(0, 30, "red channel", col="white", pos=4)
plot(as.raster(x_train[200,,,2]))
text(0, 30, "green channel", col="white", pos=4)
plot(as.raster(x_train[200,,,3]))
text(0, 30, "blue channel", col="white", pos=4)


#' Define the CNN architecture (warnings as usual)

tensorflow::set_random_seed(6957)

modcnn1 <- keras_model_sequential(input_shape=c(32,32,3)) %>%
#   1st convolution-pool layer sequence
    layer_conv_2d(filters=32, kernel_size=c(3,3), padding="same") %>%
    layer_activation_relu() %>% 
    layer_max_pooling_2d(pool_size=c(2,2)) %>%
#   2nd convolution-pool layer sequence    
    layer_conv_2d(filters=64, kernel_size=c(3,3), padding="same") %>% 
    layer_activation_relu() %>% 
    layer_max_pooling_2d(pool_size=c(2,2)) %>%
#   3rd convolution-pool layer sequence    
    layer_conv_2d(filters=128, kernel_size=c(3,3), padding="same") %>% 
    layer_activation_relu() %>% 
    layer_max_pooling_2d(pool_size=c(2,2)) %>%
#   4th convolution-pool layer sequence
    layer_conv_2d(filters=256, kernel_size=c(3,3), padding="same") %>% 
    layer_activation_relu() %>% 
    layer_max_pooling_2d(pool_size=c(2,2)) %>%
#   Flatten with dropout regularization
    layer_flatten() %>%
    layer_dropout(rate=0.5) %>%
#   Standard dense layer
    layer_dense(units=512) %>%
    layer_activation_relu() %>%
#   Output layer with softmax (61 categories to predict)    
    layer_dense(units=61) %>% 
    layer_activation_softmax()

#' Check the architecture

modcnn1

#' We see that the model has almost 1 million parameters! For example, in the
#' first convolutional layer we have 32 filters, each 3x3, for each of the 3
#' input channels (RGB), so 32 x 3 x 3 x 3 = 864 weights to which we add 32 bias
#' parameters (one for each output channel) to give 896 parameters. In the
#' second convolutional layer we have 64 x 3 x 3 x 32 + 64 = 18496, and so on.
#' At the input to the dense feedforward network where the array is flattened we
#' have 1024 nodes connected to 512 nodes, so 1024 x 512 weights + 512 biases =
#' 524800 parameters. Nevertheless, we do have a lot of data, about 94 million
#' pixels (30500 x 32 x 32 x 3).
#' 

#' Compile, and fit with an 80/20 train/validate split
#+ eval=FALSE

compile(modcnn1,
        loss="categorical_crossentropy",
        optimizer="rmsprop",
        metrics="accuracy")

fit(modcnn1, x_train, y_train,
    epochs=30,
    batch_size=128,
    validation_split=0.2) -> history

#' Save the model (or load previously trained model)

# save_model_hdf5(modcnn1, "08_3_convolutional_nnet_files/saved/modcnn1.hdf5")
# save(history, file="08_3_convolutional_nnet_files/saved/modcnn1_history.Rdata")
modcnn1 <- load_model_hdf5("08_3_convolutional_nnet_files/saved/modcnn1.hdf5")
load("08_3_convolutional_nnet_files/saved/modcnn1_history.Rdata")

#' Training history. We see evidence of overfitting after about 20 epochs as the
#' validation loss begins to climb again.

plot(history, smooth=FALSE)

#' Plot a random selection of predictions

selection <- sort(sample(1:dim(x_test)[1], 16))
par(mar=c(0,0,0,0), mfrow=c(4,4))
for ( i in selection ) {
    pred <- as.numeric(predict(modcnn1, x_test[i,,,,drop=FALSE]))
    plot(as.raster(x_test[i,,,]))
    text(0, 30, paste("prediction =", ecosubset$name[which.max(pred)]), col="red", pos=4)
    text(0, 28, paste("prob =", round(pred[which.max(pred)],2)), col="red", pos=4)
    text(0, 26, paste("actual =", ecosubset$name[y_test[i,]+1]), col="red", pos=4)
} 

#' Predictions and overall accuracy

pred_prob <- predict(modcnn1, x_test)
pred_cat <- as.numeric(k_argmax(pred_prob))
mean(pred_cat == drop(y_test))

#' Plot probabilities for the same selection of test cases as above

pred_prob %>% 
    data.frame() %>% 
    mutate(case=seq(nrow(.))) %>%
    tidyr::pivot_longer(cols=starts_with("X"), names_to="category", values_to="probability") %>% 
    mutate(category=as.integer(sub("X", "", category)) - 1) %>% 
    filter(case %in% selection) %>% 
    ggplot() +
    geom_point(aes(x=category, y=probability)) +
    facet_wrap(vars(case), nrow=4, ncol=4, labeller=label_both)
    
    


