#' ---
#' title: "Classification in Machine Learning"
#' author: Brett Melbourne
#' date: 26 Jan 2022
#' output:
#'     github_document
#' ---

#' This example is from Chapter 2.2.3 of James et al. (2021). An Introduction to
#' Statistical Learning. It is the simulated dataset in Fig 2.13.

#+ results=FALSE, message=FALSE, warning=FALSE
library(ggplot2)
library(dplyr)

#' Orange-blue data:

orbludat <-  read.csv("data/orangeblue.csv")

orbludat %>% 
    ggplot() +
    geom_point(aes(x=x1, y=x2, col=category), shape=1, size=2) +
    scale_color_manual(values=c("blue","orange")) +
    theme(panel.grid=element_blank())


#' We'll start by expanding the capability of our KNN function to handle
#' multiple x variables and the classification case with 2 categories.

# KNN function for a data frame of x_new
# x:       x data of variables in columns (matrix, numeric)
# y:       y data, 2 categories (vector, character)
# x_new:   values of x variables at which to predict y (matrix, numeric)
# k:       number of nearest neighbors to average (scalar, integer)
# return:  predicted y at x_new (vector, character)
#
knn_classify2 <- function(x, y, x_new, k) {
    category <- unique(y)
    y_int <- ifelse(y == category[1], 0, 1)
    nx <- nrow(x)
    n <- nrow(x_new)
    c <- ncol(x_new)
    p_cat2 <- rep(NA, n)
    for ( i in 1:n ) {
    #   Distance of x_new to other x (Euclidean, i.e. sqrt(a^2+b^2+..))
        x_new_m <- matrix(x_new[i,], nx, c, byrow=TRUE)
        d <- sqrt(rowSums((x - x_new_m) ^ 2))
    #   Sort y ascending by d; break ties randomly
        y_sort <- y_int[order(d, sample(1:length(d)))]
    #   Mean of k nearest y data (gives probability of category 2)
        p_cat2[i] <- mean(y_sort[1:k])
    }
    y_pred <- ifelse(p_cat2 > 0.5, category[2], category[1])
    return(y_pred)
}

#' Test the output of the knn_classify2 function.

knn_classify2(x=as.matrix(orbludat[,c("x1","x2")]),
              y=orbludat$category,
              x_new=matrix(runif(8),nrow=4,ncol=2), 
              k=4)

#' Plot. Use this block of code to try different values of k (i.e. different
#' numbers of nearest neighbors).

grid_x  <- expand.grid(x1=seq(0, 1, by=0.01), x2=seq(0, 1, by=0.01))
pred_category <- knn_classify2(x=as.matrix(orbludat[,c("x1","x2")]),
                               y=orbludat$category,
                               x_new=as.matrix(grid_x),
                               k=4)
preds <- data.frame(grid_x, category=pred_category)

orbludat %>% 
    ggplot() +
    geom_point(aes(x=x1, y=x2, col=category), shape=1, size=2) +
    geom_point(data=preds, aes(x=x1, y=x2, col=category), size=0.5) +
    scale_color_manual(values=c("blue","orange")) +
    theme(panel.grid=element_blank())


#' k-fold CV for KNN. Be careful not to confuse the k's!

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

# Function to perform k-fold CV for the KNN model algorithm on the orange-blue
# data from James et al. Ch 2.
# k_cv:    number of folds (scalar, integer)
# k_knn:   number of nearest neighbors to average (scalar, integer)
# return:  CV error as error rate (scalar, numeric)
#
cv_orblu <- function(k_cv, k_knn) {
    orbludat$fold <- random_folds(nrow(orbludat), k_cv)
    e <- rep(NA, k_cv)
    for ( i in 1:k_cv ) {
        test_data <- orbludat %>% filter(fold == i)
        train_data <- orbludat %>% filter(fold != i)
        pred_category <- knn_classify2(x=as.matrix(train_data[,c("x1","x2")]),
                               y=train_data$category,
                               x_new=as.matrix(test_data[,c("x1","x2")]),
                               k=k_knn)
        e[i] <- mean( test_data$category != pred_category )
    }
    cv_error <- mean(e)
    return(cv_error)
}

#' Test the function

cv_orblu(k_cv=10, k_knn=10)
cv_orblu(k=nrow(orbludat), k_knn=10) #LOOCV

#' Explore a grid of values for k_cv and k_knn

grid <- expand.grid(k_cv=c(5,10,nrow(orbludat)), k_knn=1:16)
cv_error <- rep(NA, nrow(grid))
set.seed(6456) #For reproducible results in this text
for ( i in 1:nrow(grid) ) {
    cv_error[i] <- cv_orblu(grid$k_cv[i], grid$k_knn[i])
    print(i/nrow(grid))
}
result1 <- cbind(grid, cv_error)


#' Plot the result.

result1 %>% 
    ggplot() +
    geom_line(aes(x=k_knn, y=cv_error, col=factor(k_cv))) +
    labs(col="k_cv")

#' LOOCV (k_cv = 199) identifies the KNN model with k_knn = 4 nearest neighbors
#' as having the best predictive performance. We see again that there is a lot
#' of variance in 5-fold and 10-fold CV and so we would not want to trust a
#' single run of k-fold CV. Repeated runs of LOOCV suggest it is essentially
#' deterministic (tied distances must be rare). Let's look at 5-fold CV with
#' many replicate CV runs with random folds.

#+ results=FALSE, cache=TRUE
grid <- expand.grid(k_cv=c(5), k_knn=1:16)
reps <- 250
cv_error <- matrix(NA, nrow=nrow(grid), ncol=reps)
set.seed(8031) #For reproducible results in this text
for ( j in 1:reps ) {
    for ( i in 1:nrow(grid) ) {
        cv_error[i,j] <- cv_orblu(grid$k_cv[i], grid$k_knn[i])
    }
    print(j) #monitor
}
result2 <- cbind(grid,cv_error)
result2$mean_cv <- rowMeans(result2[,-(1:2)])

#' Plot the result.

loocv <- result1 %>% 
    filter(k_cv == nrow(orbludat))

result2 %>%
    select(k_cv, k_knn, mean_cv) %>%
    rename(cv_error=mean_cv) %>%
    rbind(.,loocv) %>% 
    ggplot() +
    geom_line(aes(x=k_knn, y=cv_error, col=factor(k_cv))) +
    labs(title=paste("Mean across",reps,"k-fold CV runs"), col="k_cv")

#' and print out to see the detailed numbers

result2 %>% 
    select(k_cv, k_knn, mean_cv) %>%
    rename(cv_error=mean_cv) %>%
    rbind(.,loocv) %>% 
    arrange(k_cv)




