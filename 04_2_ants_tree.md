Ant data: decision trees
================
Brett Melbourne
31 Jan 2022

Decision trees for the regression case illustrated with the ants data.
We start with a regression tree for a single predictor variable, then
look at regression trees with multiple predictors.

``` r
library(ggplot2)
library(dplyr)
library(tree)
```

Forest ant data:

``` r
forest_ants <- read.csv("data/ants.csv") %>% 
    filter(habitat=="forest") %>% 
    select(latitude, richness)
```

Tree model + training algorithm. The default training algorithm has
stopping rules that include the number of observations in nodes and the
variance within nodes.

``` r
fit <- tree(richness ~ latitude, data=forest_ants)
plot(fit, type="uniform")
text(fit, pretty=0, digits=2)
```

![](04_2_ants_tree_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

We see that the tree splits latitude twice, first at 42.575 then at
42.18 to give three terminal nodes. The predicted richness shown for
each node is the mean of the data in each node.

Plot predictions with the data.

``` r
grid_latitude  <- seq(min(forest_ants$latitude), max(forest_ants$latitude), length.out=201)
grid_data <- data.frame(latitude=grid_latitude)
preds <- cbind(grid_data, richness=predict(fit, newdata=grid_data))

forest_ants %>% 
    ggplot() +
    geom_point(aes(x=latitude, y=richness)) +
    geom_line(data=preds, aes(x=latitude, y=richness)) +
    coord_cartesian(ylim=c(0,20))
```

![](04_2_ants_tree_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

We see that for a single predictor variable, a regression tree
partitions the predictor (x axis) into segments (three segments in this
case).

In contrast, the following fit is for a deeper tree. We have modified
the stopping rules of the training algorithm to allow splits all the way
to individual data points.

``` r
fit <- tree(richness ~ latitude, data=forest_ants, 
            control=tree.control(nobs=nrow(forest_ants), minsize=2, mindev=0))
plot(fit, type="uniform")
text(fit, pretty=0, digits=2)
```

![](04_2_ants_tree_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

Plot predictions with the data.

``` r
grid_latitude  <- seq(min(forest_ants$latitude), max(forest_ants$latitude), length.out=201)
grid_data <- data.frame(latitude=grid_latitude)
preds <- cbind(grid_data, richness=predict(fit, newdata=grid_data))

forest_ants %>% 
    ggplot() +
    geom_point(aes(x=latitude, y=richness)) +
    geom_line(data=preds, aes(x=latitude, y=richness)) +
    coord_cartesian(ylim=c(0,20))
```

![](04_2_ants_tree_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

For this tree the predictions follow the data except for the one case
where two data points shared the same latitude.

There is more data in the ants dataset, including two more predictor
variables: habitat (bog or forest) and elevation (m).

``` r
ants <- read.csv("data/ants.csv") %>% 
    select(-site) %>% 
    mutate_if(is.character, factor)
head(ants)
```

    ##   habitat latitude elevation richness
    ## 1  forest    41.97       389        6
    ## 2  forest    42.00         8       16
    ## 3  forest    42.03       152       18
    ## 4  forest    42.05         1       17
    ## 5  forest    42.05       210        9
    ## 6  forest    42.17        78       15

Fit a tree that includes both latitude and habitat as predictors

``` r
fit <- tree(richness ~ latitude + habitat, data=ants)
plot(fit, type="uniform")
text(fit, pretty=0, digits=2)
```

![](04_2_ants_tree_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

We see the tree has nodes that split at both predictor variables. First
it splits by latitude, then it splits by habitat, then it splits by
latitude again. At the habitat nodes, bog is to the left while forest is
to the right.

Plot the prediction from the fitted model

``` r
grid_data  <- expand.grid(
    latitude=seq(min(ants$latitude), max(ants$latitude), length.out=201),
    habitat=factor(c("forest","bog")))
preds <- cbind(grid_data, richness=predict(fit, newdata=grid_data))

ants %>% 
    ggplot() +
    geom_point(aes(x=latitude, y=richness, col=habitat)) +
    geom_line(data=preds, aes(x=latitude, y=richness, col=habitat)) +
    coord_cartesian(ylim=c(0,20))
```

![](04_2_ants_tree_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

Plotting the predicted richness reveals that we have effectively modeled
richness as a nonlinear combination, or “interaction”, of habitat and
latitude. The stepwise functions broadly (arguably crudely) capture the
pattern of a different nonlinear relationship between richness and
latitude in each habitat.

Now fit a tree with all three predictor variables

``` r
fit <- tree(richness ~ latitude + habitat + elevation, data=ants)
plot(fit, type="uniform")
text(fit, pretty=0, digits=2)
```

![](04_2_ants_tree_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

This tree has splits for all three predictor variables. First it splits
by latitude, then by habitat, and then it splits again by elevation in
forest habitat only. One interpretation is that elevation is important
to predict richness only in forest.

It’s harder to visualize the prediction from this fit since we have
multiple predictor dimensions. Here is one visualizaton:

``` r
# First make some midpoints for three levels of elevation and get predictions
# for these in combination with habitat and latitude
low <- round(mean(c(min(ants$elevation),181)))
mid <- round(mean(c(181,318)))
high <- round(mean(c(max(ants$elevation),318)))
grid_data  <- expand.grid(
    latitude=seq(min(ants$latitude), max(ants$latitude), length.out=201),
    habitat=factor(c("forest","bog")),
    elevation=c(low,mid,high))
preds <- cbind(grid_data, richness=predict(fit, newdata=grid_data))
preds$elev_gp <- factor(preds$elevation, labels=c("low","mid","high"))

# It's some work, with lots of manual manipulation, to make a clear plot
ants %>%
    mutate(elev_gp=cut(elevation, 
                       breaks=c(0,181,318,545), 
                       labels=c("low","mid","high"))) %>% 
    ggplot() +
    geom_point(aes(x=latitude, y=richness, col=habitat, shape=elev_gp)) +
    geom_line(data=filter(preds, habitat=="bog"), 
              aes(x=latitude, y=richness), col="#F8766D", size=1) +
    geom_line(data=filter(preds, habitat=="forest", elev_gp=="mid"), 
              aes(x=latitude, y=richness), col="black", size=1) +
    geom_line(data=filter(preds, habitat=="forest", elev_gp=="low"),
              aes(x=latitude, y=richness), col="#00BFC4", size=1, linetype=2) +
    geom_line(data=filter(preds, habitat=="forest", elev_gp=="high"), 
              aes(x=latitude, y=richness), col="#C77CFF", size=1, linetype=2) +
    geom_text(aes(x=43.7, y=2.8, label="Bog"), col="#F8766D", hjust="left") +
    geom_text(aes(x=43.2, y=4.6, label="Forest: high"), col="#C77CFF", hjust="left") +
    geom_text(aes(x=42.9, y=7.3, label="Forest: mid"), col="black", hjust="left") +
    geom_text(aes(x=42.6, y=12.5, label="Forest: low"), col="#00BFC4", hjust="left") +
    coord_cartesian(ylim=c(0,20))
```

![](04_2_ants_tree_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

The plot shows that we’re modeling different nonlinearities in the
forest habitat for low, medium and high elevations. Effectively we are
crudely modeling the interaction between habitat, latitude, and
elevation. Since our goal is prediction we’d be most interested in
plotting predictions within some region to make a map of predicted
species richness for the area from which the data came. For that we’d
need maps of the predictor variables but such a visualization scales to
any number of predictor variables.

The prediction error (i.e. the out-of-sample error) from a regression
tree can be estimated by k-fold cross validation in the usual way.

``` r
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

# Function to perform k-fold CV for the tree model on ants data
# k:       number of folds (scalar, integer)
# return:  CV error as MSE (scalar, numeric)
#
cv_ants <- function(k) {
    ants$fold <- random_folds(nrow(ants), k)
    e <- rep(NA, k)
    for ( i in 1:k ) {
        test_data <- ants %>% filter(fold == i)
        train_data <- ants %>% filter(fold != i)
        train_tree <- tree(richness ~ latitude + habitat + elevation, data=train_data)
        pred_richness <- predict(train_tree, newdata=test_data)
        e[i] <- mean((test_data$richness - pred_richness) ^ 2)
    }
    cv_error <- mean(e)
    return(cv_error)
}
```

Test the function

``` r
cv_ants(k=5)
```

    ## [1] 10.70025

``` r
cv_ants(k=nrow(ants)) #LOOCV
```

    ## [1] 12.68253

Running the above two lines of code multiple times we find lots of
variability in the prediction error estimate for 5-fold CV due to the
randomness of the folds. LOOCV does not change because the tree model
and training algorithms are deterministic. As before, we’ll need
repeated folds for a more stable estimate of the 5-fold CV:

``` r
set.seed(3127)
reps <- 500
cv_error <- rep(NA, reps)
for ( i in 1:reps ) {
    cv_error[i] <- cv_ants(k=5)
}
```

A histogram suggests the CV replicates are well behaved

``` r
hist(cv_error)
```

![](04_2_ants_tree_files/figure-gfm/unnamed-chunk-15-1.png)<!-- -->

Estimated error and its Monte Carlo error (about +/- 0.1)

``` r
mean(cv_error)
```

    ## [1] 13.15425

``` r
sd(cv_error) / sqrt(reps)
```

    ## [1] 0.08818662
