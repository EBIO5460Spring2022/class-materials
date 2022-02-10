### Week 4 homework

Due Wed 16 Feb

#### 1. Coding: classification trees

The goal of this homework is to compare the predictive performance (using k-fold CV) of decision-tree models to the KNN model from the previous homework for the presence/absence of the Willow Tit.

* Start with a simple classification tree. Use the `tree()` function from the `tree` package. The option `type="class" ` in `predict()` will return the predicted class (see `?predict.tree`).
* Modify the `bagrt()` function to make a `bagct()` function for bagged classification trees. Whereas for bagged regression trees we averaged the prediction for a numerical response variable (e.g. richness in the ants example) across trees, for bagged classification trees we need to average the probabilities across trees before making the final present/absent prediction. The option `type="vector" ` in `predict()` will return the probabilities (see `?predict.tree`).
* Use `bagct()` on the Willow Tit data.
* Use the k-fold CV inference algorithm to compare KNN, simple tree, and bagged tree models for the Willow Tit dataset.
* Optional advanced: So far we have plotted presence/absence as a map. Modify the code for your best performing model to output the probabilities instead of presence/absence class. Plot the probabilities as a map.

**Push your code to GitHub**


#### 2. Reading
If you didn't already read this section in preparation for the cancelled Wed class, please do: James et al. Chapter 8.2: Bagging, Random Forests and Boosting.

