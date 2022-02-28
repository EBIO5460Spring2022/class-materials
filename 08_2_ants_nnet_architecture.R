#' ---
#' title: "Ant data: neural network architectures"
#' author: Brett Melbourne
#' date: 28 Feb 2022
#' output:
#'     github_document
#' ---

#' Different neural network architectures illustrated with the
#' ants data using Keras.

#+ results=FALSE, message=FALSE, warning=FALSE
library(ggplot2)
library(dplyr)
library(keras)

#' Ant data with 3 predictors of species richness

ants <- read.csv("data/ants.csv") %>% 
    select(richness, latitude, habitat, elevation) %>% 
    mutate(habitat=factor(habitat))

#' Prepare the data and a set of new x to predict
 
xtrain <- ants[,-1] %>% 
    mutate(across(where(is.numeric), scale)) %>% 
    model.matrix(~ . -1, .)

ytrain <- ants[,1]

grid_data  <- expand.grid(
    latitude=seq(min(ants$latitude), max(ants$latitude), length.out=201),
    habitat=factor(c("forest","bog")),
    elevation=seq(min(ants$elevation), max(ants$elevation), length.out=51))

x <- grid_data %>% 
    mutate(across(where(is.numeric), scale)) %>% 
    model.matrix(~ . -1, .)


#' A wide model with 25 units

tensorflow::set_random_seed(6590)
modnn2 <- keras_model_sequential(input_shape = ncol(xtrain)) %>%
    layer_dense(units = 25) %>%
    layer_activation("relu") %>% 
    layer_dense(units = 1)
compile(modnn2, optimizer="rmsprop", loss="mse")
fit(modnn2, xtrain, ytrain, epochs = 500, batch_size=4) -> history
# save_model_hdf5(modnn2, "08_2_ants_nnet_architecture_files/saved/modnn2.hdf5")
# save(history, file="08_2_ants_nnet_architecture_files/saved/modnn2_history.Rdata")
modnn2 <- load_model_hdf5("08_2_ants_nnet_architecture_files/saved/modnn2.hdf5")
load("08_2_ants_nnet_architecture_files/saved/modnn2_history.Rdata")
plot(history, smooth=FALSE, theme_bw=TRUE)

npred <- predict(modnn2, x)
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

#' A deep model with 25 units

tensorflow::set_random_seed(7855)
modnn3 <- keras_model_sequential(input_shape = ncol(xtrain)) %>%
    layer_dense(units = 5) %>%
    layer_activation("relu") %>%
    layer_dense(units = 5) %>%
    layer_activation("relu") %>% 
    layer_dense(units = 5) %>%
    layer_activation("relu") %>% 
    layer_dense(units = 5) %>%
    layer_activation("relu") %>% 
    layer_dense(units = 5) %>%
    layer_activation("relu") %>% 
    layer_dense(units = 1)
compile(modnn3, optimizer="rmsprop", loss="mse")
fit(modnn3, xtrain, ytrain, epochs = 500, batch_size=4) -> history
# save_model_hdf5(modnn3, "08_2_ants_nnet_architecture_files/saved/modnn3.hdf5")
# save(history, file="08_2_ants_nnet_architecture_files/saved/modnn3_history.Rdata")
modnn3 <- load_model_hdf5("08_2_ants_nnet_architecture_files/saved/modnn3.hdf5")
load("08_2_ants_nnet_architecture_files/saved/modnn3_history.Rdata")
plot(history, smooth=FALSE, theme_bw=TRUE)

npred <- predict(modnn3, x)
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


