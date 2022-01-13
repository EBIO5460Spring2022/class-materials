### Week 1 homework

Setup GitHub, read about predictive performance and cross-validation, code the k-fold CV algorithm and explore predictive performance for a smoothing-spline model with the ants data.

#### 1. Take control of your GitHub repo for this class

First, email me your GitHub username so I can add you to the class organization. Once I've done that, you'll be able to see the class repositories. I have set up a GitHub repo for you that is within the private GitHub space for this class. This repo is not public (i.e. not open to the world). You and I both have write access to this repo. Clone it to your computer using RStudio.

1. Go to the class GitHub organization. From your GitHub profile page, look to the left for organizations. The organization name is EBIO5460Spring2022. Or directly, the URL is https://github.com/EBIO5460Spring2022.
2. Find your repo. It is called ml4e_firstnamelastinitial.
3. From the green `Code` button in your repo, copy its URL to the clipboard.
4. Clone it to an RStudio project on your computer.
   1. File > New Project > Version Control > Git.
   2. In “Repository URL”, paste the URL


In the repository you just cloned, you will find three files.

1. A file with extension `.Rproj`. This file was created by RStudio. To open your project in RStudio, double click the file's icon. When RStudio opens, you will be in the working directory for the project.
2. `README.md`. This file was created by GitHub. This is a plain text file with extension `.md`, indicating that it is a file in Markdown format. You can edit this file using a text editor.
3. `.gitignore`. This file was created by RStudio but it is a file used by Git. This file tells Git which files or types of files to ignore in your repository (i.e. files that will not be under version control). By default, RStudio tells Git to ignore several files including `.Rhistory` and `.Rdata` because it usually doesn't make sense to track these temporary files. You can use a text editor to add other files to `.gitignore`.

If you need a git refresher, you can use the tutorials from fall 2021 [here](https://github.com/EBIO5460Fall2021/class-materials/tree/main/skills_tutorials).

#### 2. Reading

James et al. (2021) An Introduction to Statistical Learning

* Foundations: pp 15-19 (but not the section titled "inference")
* Assessing model accuracy: pp 28-42.
* Cross validation: pp 198-208.

#### 3. Coding

Investigate cross-validation with ants data and a smoothing model

``` r
library(ggplot2)
library(dplyr)
```



Forest ant data. It's in the class-materials repository in the `data` folder, which you can replicate in your repository.

``` r
forest_ants <- read.csv("data/ants.csv") %>% 
    filter(habitat=="forest")

forest_ants %>% 
    ggplot() +
    geom_point(aes(x=latitude, y=richness)) +
    ylim(0,20)
```



Example of a smoothing spline model. Try running this next block of code to visualize the model predictions for different values of `df`.

```r
fit <- smooth.spline(forest_ants$latitude, forest_ants$richness, df=7)
xx  <- seq(min(forest_ants$latitude), max(forest_ants$latitude), length.out=201)
preds <- data.frame(predict(fit, xx))
forest_ants %>% 
    ggplot() +
    geom_point(aes(x=latitude, y=richness)) +
    geom_line(data=preds, aes(x=x, y=y)) +
    coord_cartesian(ylim=c(0,20))
```



Using `predict` to ask for predictions from the fitted smoothing-spline model.

```r
predict(fit, x=43.2)
predict(fit, x=forest_ants$latitude)
```



Making folds for k-fold CV

```r
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
```



What does the output of `random_folds()` look like?

```
random_folds(nrow(forest_ants), k=5)
random_folds(nrow(forest_ants), k=nrow(forest_ants))
```



Now code up the k-fold CV algorithm to estimate the prediction mean squared error for one value of df. Try 5-fold, 10-fold, and n-fold. Try different values of df.

```r
# k-fold CV algorithm
# divide dataset into k parts i = 1...k
# initiate vector to hold e
# for each i
#     test dataset = part i
#     training dataset = remaining data
#     find f using training dataset
#     use f to predict for test dataset
#     e_i = prediction error (mse)
# CV_error = mean(e)
```



**Push your code to GitHub**
