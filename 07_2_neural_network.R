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
    mutate(hab_forest=ifelse(habitat=="forest", 1, 0)) %>% 
    mutate(hab_bog=ifelse(habitat=="bog", 1, 0)) %>%
    select(where(is.numeric)) %>%                 #drop original categorical var
    as.matrix()

# dimensions of x
n <- nrow(x)
p <- ncol(x)

# add intercept column to x to form the model matrix
mmx <- cbind(rep(1, n), x)

# set K
K <- 10

# set parameters
w <- rnorm(20) %>% matrix(nrow=K, ncol=p+1)
beta <- rnorm(5) %>% matrix(nrow=K+1, ncol=1)

w <- w_hat
beta <- beta_hat

# hidden layer, iterating over each activation unit
A <- matrix(NA, nrow=n, ncol=K)
for ( k in 1:K ) {
#   linear predictor (via model matrix)
    z <- mmx %*% t(w[k,,drop=FALSE])
#   nonlinear activation
    A[,k] = g_relu(z)
}

# output layer, linear model (via model matrix)
mmA <- cbind(rep(1, n), A) #model matrix for A
f_x <- mmA %*% beta

# return f(x)
nn1_preds <- f_x

#' In the code above, we twice constructed a model matrix for use in the linear
#' algebra steps. We could instead have used the R-centric:

#+ eval=FALSE
model.matrix(~ ., x)

#' but I find the code above more portable (i.e. the algorithm is easily
#' rewritten for another language) and more literal for understanding the matrix
#' workflow. We also hand-constructed the dummy variables and we could do that
#' more conveniently with `model.matrix`, where the formula expression drops the
#' intercept term to fully expand the categorical variable:

#+ eval=FALSE
x <- grid_data %>% 
    mutate(across(where(is.numeric), scale)) %>% 
    model.matrix(~ . -1, .)

#' We would still need to add an intercept column to form a full model matrix
#' for the linear algebra step. We could do it all in one piped operation with
#' two calls to `model.matrix` and a dataframe conversion in between but this is
#' obviously fairly obtuse:
#' 

#+ eval=FALSE
mmx <- grid_data %>% 
    mutate(across(where(is.numeric), scale)) %>% 
    model.matrix(~ . -1, .) %>% 
    data.frame() %>% 
    model.matrix(~ ., .)


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
    mutate(hab_forest=ifelse(habitat=="forest", 1, 0)) %>% 
    mutate(hab_bog=ifelse(habitat=="bog", 1, 0)) %>%
    select(where(is.numeric)) %>%                 #drop original categorical var
    as.matrix()
ytrain <- ants[,1]

modnn <- keras_model_sequential() %>%
    layer_dense(units = 10, activation = "relu", 
                input_shape = ncol(x)) %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 1)

modnn %>% compile(loss = "mse",
                  optimizer = optimizer_rmsprop(),
                  metrics = list("mean_absolute_error"))
                  
summary(modnn)

fitnn <- fit(modnn, xtrain, ytrain, epochs = 800, batch_size=44)

plot(fitnn)

npred <- predict(modnn, x)
summary(fitnn)
coef(modnn)

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

wts <- get_weights(modnn)
w_hat <- t(rbind(wts[[2]], wts[[1]]))
beta_hat <- t(t(c(wts[[4]], wts[[3]])))
