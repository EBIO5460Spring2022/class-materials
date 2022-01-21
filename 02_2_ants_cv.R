#' ---
#' title: "Ant data: k-fold cross validation"
#' author: Brett Melbourne
#' date: 19 Jan 2022
#' output:
#'     github_document
#' ---

#' Investigate cross-validation with the ants data and a smoothing-spline model

#+ results=FALSE, message=FALSE, warning=FALSE
library(ggplot2)
library(dplyr)
library(tidyr)

#' Forest ant data:

forest_ants <- read.csv("data/ants.csv") %>% 
    filter(habitat=="forest")

forest_ants %>% 
    ggplot() +
    geom_point(aes(x=latitude, y=richness)) +
    coord_cartesian(ylim=c(0,20))

#' Example of a smoothing spline model. Try running this next block of code to
#' visualize the model predictions for different values of `df`. Here is df=7.

fit <- smooth.spline(forest_ants$latitude, forest_ants$richness, df=7)
x_grid  <- seq(min(forest_ants$latitude), max(forest_ants$latitude), length.out=201)
preds <- data.frame(predict(fit, x=x_grid))
forest_ants %>% 
    ggplot() +
    geom_point(aes(x=latitude, y=richness)) +
    geom_line(data=preds, aes(x=x, y=y)) +
    coord_cartesian(ylim=c(0,20))

#' Using `predict` to ask for predictions from the fitted smoothing spline
#' model.

predict(fit, x=43.2)
predict(fit, x=forest_ants$latitude)
predict(fit, x=seq(41, 45, by=0.5))

#' Implement the k-fold CV algorithm. First we need a function to create the
#' folds. It needs to deal with the common case that the data can't be divided
#' into folds of exactly equal size. Some folds will have an extra data point.

# Function to partition a data set into random folds for cross-validation
# n:       length of dataset (scalar, integer)
# k:       number of folds (scalar, integer)
# return:  fold labels (vector, integer)
# 
random_folds <- function(n, k) {
    min_n <- floor(n / k)
    extras <- n - k * min_n
    labels <- c(rep(1:k, each=min_n),rep(seq_len(extras)))
    folds <- sample(labels, n)
    return(folds)
}

#' What is the output of `random_folds()`?

random_folds(nrow(forest_ants), k=5)
random_folds(nrow(forest_ants), k=nrow(forest_ants)) #k=n is LOOCV

#' Now code up the k-fold CV algorithm to estimate the prediction mean squared
#' error for one value of df, translating from our pseudocode to R. We can run
#' this block of code to try different values of df and to try 5-fold, 10-fold,
#' and n-fold CV.

df <- 7
k <- 5

# divide dataset into k parts i = 1...k
forest_ants$fold <- random_folds(nrow(forest_ants), k)

# initiate vector to hold e
e <- rep(NA, k)

# for each i
for ( i in 1:k ) {
#   test dataset = part i
    test_data <- forest_ants %>% filter(fold == i)
#   training dataset = remaining data
    train_data <- forest_ants %>% filter(fold != i)
#   find f using training dataset
    trained_f <- smooth.spline(train_data$latitude, train_data$richness, df=df)
#   use f to predict for test dataset
    pred_richness <- predict(trained_f, test_data$latitude)$y
#   e_i = prediction error (mse)
    e[i] <- mean((test_data$richness - pred_richness) ^ 2)
}

# CV_error = mean(e)
cv_error <- mean(e)
cv_error


#' To help us do some systematic experiments to explore different combinations
#' of df and k we can encapsulate the above as a function.

# Function to perform k-fold CV for a smoothing spline on ants data
# k:       number of folds (scalar, integer)
# df:      degrees of freedom in smoothing spline (scalar, integer)
# return:  CV error as RMSE (scalar, numeric)
#
cv_ants <- function(k, df) {
    forest_ants$fold <- random_folds(nrow(forest_ants), k)
    e <- rep(NA, k)
    for ( i in 1:k ) {
        test_data <- forest_ants %>% filter(fold == i)
        train_data <- forest_ants %>% filter(fold != i)
        trained_f <- smooth.spline(train_data$latitude, train_data$richness, df=df)
        pred_richness <- predict(trained_f, test_data$latitude)$y
        e[i] <- mean((test_data$richness - pred_richness) ^ 2)
    }
    cv_error <- mean(e)
    return(cv_error)
}

#' Test the function

cv_ants(k=5, df=7) 
cv_ants(k=nrow(forest_ants), df=7)

#' Explore a grid of values for df and k

grid <- expand.grid(k=c(5,10,nrow(forest_ants)), df=2:16)
grid

cv_error <- rep(NA, nrow(grid))
set.seed(1280) #For reproducible results in this text
for ( i in 1:nrow(grid) ) {
    cv_error[i] <- cv_ants(grid$k[i], grid$df[i])
}
result1 <- cbind(grid,cv_error)

#' Plot the result. Make k a factor to get discrete colors.

result1 %>% 
    ggplot() +
    geom_line(aes(x=df, y=cv_error, col=factor(k)))

#' We see that RMSE prediction error (cv_error) increases dramatically for df
#' beyond 8 or so. We also see that cv_error estimates are quite variable for
#' k=10 and especially k=5. This is due to the randomness of partitioning a very
#' small dataset into folds. If we repeat the above with a different seed, we'd
#' get different results for k=5 or k=10. LOOCV is deterministic, so it won't
#' differ if we repeat it. Let's zoom in to the zone with the best performing
#' values for df.
    
result1 %>% 
    ggplot() +
    geom_line(aes(x=df, y=cv_error, col=factor(k))) +
    coord_cartesian(xlim=c(2,8), ylim=c(10,25))

#' LOOCV (k=22) identifies df=3 as the best performing model, whereas in this
#' particular run 10-fold CV identifies df=4 and 5-fold CV identifies df=6. What
#' should we do here? Given the uncertainty in RMSE estimates for the lower
#' values of k, we'd be best to use LOOCV as a default. But we could also try
#' for a better estimate by repeated k-fold runs. Let's explore the variability
#' in 10-fold and 5-fold CV.

#+ results=FALSE, cache=TRUE
grid <- expand.grid(k=c(5,10), df=2:8)
reps <- 100
cv_error <- matrix(NA, nrow=nrow(grid), ncol=reps)
set.seed(1978) #For reproducible results in this text
for ( j in 1:reps ) {
    for ( i in 1:nrow(grid) ) {
        cv_error[i,j] <- cv_ants(grid$k[i], grid$df[i])
    }
    print(j) #monitor progress
}
result2 <- cbind(grid,cv_error)

#' Plot the first 10 reps for each k-fold

result2 %>% 
    select(1:12) %>%
    mutate(k=paste(k, "-fold CV", sep="")) %>%
    pivot_longer(cols="1":"10", names_to="rep", values_to="cv_error") %>% 
    mutate(rep=as.numeric(rep)) %>% 
    ggplot() +
    geom_line(aes(x=df, y=cv_error, col=factor(rep))) +
    facet_wrap(vars(k)) +
    coord_cartesian(xlim=c(2,8),ylim=c(10,25))

#' We see again that there is more variability for 5-fold CV and for both 5-fold
#' and 10-fold CV there is so much variability, we'd pick a different value for
#' df on each run. So, we wouldn't want to rely on a single k-fold run.
#' Averaging across runs would give a better estimate of the prediction RMSE.

result2$mean_cv <- rowMeans(result2[,-(1:2)])

#' Plotting the results shows that averaged across runs, we'd pick the same df
#' as LOOCV (k=22).

loocv <- result1 %>% 
    filter(k == 22, df <= 8)

result2 %>%
    select(k, df, mean_cv) %>%
    rename(cv_error=mean_cv) %>%
    rbind(.,loocv) %>% 
    ggplot() +
    geom_line(aes(x=df, y=cv_error, col=factor(k))) +
    labs(title=paste("Mean across",reps,"k-fold CV runs"), col="k")
