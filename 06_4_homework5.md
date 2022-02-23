### Week 6 homework (HW #5)

Due Mon 28 Feb

#### 1. Reading

James et al. Chapter 10: Deep learning

* Mon 21 Feb: 10.1 single layer neural networks
* Wed 23 Feb: 10.2-10.3 multilayer networks and CNNs


#### 2. Coding: boosting
The goal of this homework is to train and tune a gradient boosting model to predict the occurrence of a plant in New Zealand based on a range of predictor variables. The plant is "nz05" from an anonymized reference dataset (apparently NZ plants have better data-privacy protections than US citizens). These data are from one of the papers we'll read: 

* Valavi et al. (2021). Predictive performance of presence-only species distribution models: a benchmark study with reproducible code. *Ecological Monographs* 0:e01486. https://doi.org/10.1002/ecm.1486.

and I think this could be super fun to expand on in the later part of the semester by seeing if we can improve the predictions in this paper.

```R
library(disdat) #data package
library(dplyr)
library(ggplot2)
library(gbm)
```

Load presence-absence data for species "nz05"

```R
nz05df <- bind_cols(select(disPa("NZ"), nz05), disEnv("NZ")) %>% 
    rename(occ=nz05)
head(nz05df)
```

Outline of New Zealand

```R
nzpoly <- disBorder("NZ")
class(nzpoly) #sf = simple features; common geospatial format
```

Plot presence (1) absence (0) data

```R
nz05df %>% 
    arrange(occ) %>% #place presences on top
    ggplot() +
    geom_sf(data=nzpoly, fill="lightgray") +
    geom_point(aes(x=x, y=y, col=factor(occ)), shape=1, alpha=0.2) +
    theme_void()
```

Data for modeling

```R
nz05pa <- nz05df %>% 
    select(!c(group,siteid,x,y,toxicats)) %>% 
    mutate(age=factor(age))
head(nz05pa)
```

Example boosted model (30-60 secs)

```R
# Train
nz05_train <- gbm(occ ~ ., data=nz05pa, distribution="bernoulli", n.trees=10000, 
                  interaction.depth=1, shrinkage=0.01, bag.fraction=1)
summary(nz05_train)

# Predict
nz05_prob <- predict(nz05_train, type="response")
nz05_pred <- 1 * (nz05_prob > 0.5)

# Characteristics of this prediction
hist(nz05_prob)
max(nz05_prob)
sum(nz05_prob > 0.5) #number of predicted presences

table(nz05_pred, nz05pa$occ)  #confusion matrix
mean(nz05_pred == nz05pa$occ) #accuracy
mean(nz05_pred != nz05pa$occ) #error = 1 - accuracy
```

Train and tune a gradient boosting model to find the best predictive performance across the three boosting parameters (`interaction.depth`, `shrinkage`, `n.trees`). Leave `bag.fraction` set to 1 (<1 gives stochastic gradient boosting, which we'll explore later).

Suggestions:

* Use 5-fold CV
* Use parallel processing
* For `n.trees` you don't need to include it in a parameter grid. Instead, train the model once with a large value for `n.trees` and access the predictions for the lower values using the `n.trees` argument to `predict` 
  * e.g. `predict(nz05_train, type="response", n.trees=n)`
* Assess the model with the error rate as the loss function, i.e.
  * `mean(nz05_pred != nz05pa$occ)`

Plot the prediction both as a presence map and a map of probabilities. You can download a complete set of predictors as raster files for New Zealand from here:

* https://osf.io/kwc4v/files/
* navigate to data/Environment > click on NZ > download as zip

**Push your code to GitHub**

