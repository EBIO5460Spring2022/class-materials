#' ---
#' title: "Ant data: neural network"
#' author: Brett Melbourne
#' date: 15 Feb 2022
#' output:
#'     github_document
#' ---

#' A single layer neural network illustrated with the ants data.
#' 

#' You'll need to install Python and Tensorflow so you can use the `keras`
#' package. It's easiest to do that by carefully following the directions from
#' the [ISLR2 book
#' website](https://hastie.su.domains/ISLR2/keras-instructions.html). This
#' should be relatively painless and not disruptive to your current setup (e.g.
#' projects relying on Python or Tensorflow) but check the preamble in the
#' directions. It will set up an instance of Miniconda that will isolate Python
#' and Tensorflow from other versions on your computer. On Windows 10, the
#' default installation is into /Users/<username>/Documents/islr-miniconda and a
#' new start menu item is added called Miniforge3.

#+ results=FALSE, message=FALSE, warning=FALSE
library(ggplot2)
library(dplyr)
library(keras)

#' Ant data with 3 predictors of species richness

ants <- read.csv("data/ants.csv") %>% 
    select(richness, latitude, habitat, elevation) %>% 
    mutate(habitat=factor(habitat))

#' Single layer neural network, model algorithm:
#' ```
#' define g(z)
#' load x (in appropriate form)
#' set K
#' set parameters: w_kj, beta_k
#' for each activation unit k in 1:K
#'     calculate linear predictor: z_k = w_k0 + sum_j(w_kj x_j)
#'     calculate nonlinear activation: A_k = g(z_k)
#' calculate linear model: f(x) = beta_0 + sum_k(beta_k A_k)
#' return f(x)
#' ```
#' 

#' Code this algorithm in R. At first, we'll make a fairly literal translation,
#' except for introducing some convenient linear algebra to do the linear model
#' steps. Soon we'll take even further advantage of linear algebra to create an
#' elegant "tensor flow".
#' 

#+ cache=TRUE, results=FALSE

# Single layer neural network, model algorithm

# define g(z)
#
g_relu <- function(z) {
    g_z <- ifelse(z < 0, 0, z)  
    return(g_z)
}

# load x (could be a grid of new predictor values or the original data)
grid_data  <- expand.grid(
    latitude=seq(min(ants$latitude), max(ants$latitude), length.out=201),
    habitat=factor(c("forest","bog")),
    elevation=seq(min(ants$elevation), max(ants$elevation), length.out=51))

# data preparation: scale, dummy encoding, convert to matrix
x <- grid_data %>% 
    mutate(across(where(is.numeric), scale)) %>% 
    model.matrix(~ . -1, .)

# dimensions of x
n <- nrow(x)
p <- ncol(x)

# set K
K <- 5

# set parameters
w1 <- c(-0.96660703,  0.10571498,  0.2010671, -0.86535341,
        -0.30506051, -0.09064804,  0.7194443,  0.21766625,
         0.03310419, -0.07996691,  0.2330788, -0.00206342,
        -0.55194455,  0.34334779,  1.6975850, -0.73369455,
        -0.32670146,  0.26837730, -0.7647074,  0.57788116) %>%
    matrix(nrow=5, ncol=4, byrow=TRUE)

b1 <- c(0.1678398, 0.9261078, -0.2951148, 0.6162176, -0.3971323)

w2 <- c(0.4636577, 1.590001, -0.706589, 1.67491, -0.4370111) %>% 
    matrix(nrow=1, ncol=5, byrow=TRUE)

b2 <- 1.312699

# hidden layer 1, iterating over each activation unit
A <- matrix(NA, nrow=n, ncol=K)
for ( k in 1:K ) {
#   linear predictor (via model matrix)
    z <- x %*% t(w1[k,,drop=FALSE]) + b1[k]
#   nonlinear activation
    A[,k] = g_relu(z)
}

# output layer 2, linear model
f_x <- A %*% t(w2) + b2

# return f(x)
nn1_preds <- f_x

#' Plot predictions

preds <- cbind(grid_data, richness=nn1_preds)
ants %>% 
    ggplot() +
    geom_line(data=preds, 
              aes(x=latitude, y=richness, col=elevation, group=factor(elevation)),
              linetype=2) +
    geom_point(aes(x=latitude, y=richness, col=elevation)) +
    facet_wrap(vars(habitat)) +
    scale_color_viridis_c() +
    theme_bw()


#' With Keras
 
xtrain <- ants[,-1] %>% 
    mutate(across(where(is.numeric), scale)) %>% 
    model.matrix(~ . -1, .)

ytrain <- ants[,1]

modnn <- keras_model_sequential() %>%
    layer_dense(units = 5, activation = "relu", 
                input_shape = ncol(x)) %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 1)

modnn %>% compile(loss = "mse",
                  optimizer = optimizer_rmsprop(),
                  metrics = list("mean_absolute_error"))
                  
summary(modnn)

fitnn <- fit(modnn, xtrain, ytrain, epochs = 800, batch_size=32)
plot(fitnn)
get_weights(modnn)


npred <- predict(modnn, x)

preds <- cbind(grid_data, richness=npred)
ants %>% 
    ggplot() +
    geom_line(data=preds, 
              aes(x=latitude, y=richness, col=elevation, group=factor(elevation)),
              linetype=2) +
    geom_point(aes(x=latitude, y=richness, col=elevation)) +
    facet_wrap(vars(habitat)) +
    scale_color_viridis_c() +
    theme_bw()


