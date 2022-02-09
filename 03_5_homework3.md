### Week 3 homework

Code a classification problem. Reading about decision tree algorithms.

#### 1. Coding: classification

The goal of this homework is to find the best performing KNN model to predict the presence or absence of the Willow Tit, a bird native to Switzerland, using elevation as a predictor. This data set is from Royle JA & Dorazio RM (2008) Hierarchical Modeling and Inference in Ecology, p 87. Elevation (altitude) data are from eros.usgs.gov.

```R
library(ggplot2)
library(dplyr)

# Read in the data()
willowtit <- read.csv("data/wtmatrix.csv") %>% 
    mutate(status=ifelse(y.1==1, "present", "absent")) %>% 
    select(status, elev)
head(willowtit)

# Summary
willowtit %>% 
    group_by(bin=cut(elev, breaks=seq(0, 3000, by=500)), status) %>% 
    summarize(p=n()) 
```

Next find the best predictive KNN model by repurposing the previous KNN classification code.

Finally, make a map of where the Willow Tit is predicted to be. To get you started, here is code to read in and plot elevation data (digital elevation model) for Switzerland.

```R
swissdem <- read.csv("data/switzerland_tidy.csv")

ggplot(swissdem) +
    geom_raster(aes(x=x, y=y, fill=Elev_m)) +
    scale_fill_gradientn(colors=terrain.colors(22), name="Elevation (m)") + 
    coord_quickmap() +
    labs(title="Switzerland: DEM") +
    theme_void() +
    theme(plot.title=element_text(hjust=0.5, vjust=-2))
```

Here is how we could overlay the presence of the Willow Tit on the DEM. First add a column to `swissdem` indicating presence/absence. To provide working code for illustration only, I've hard coded elevations here but you would instead add a column of model predictions (e.g. using `cbind()`). 

```R
swissdem <- swissdem %>% 
    mutate(present=ifelse(Elev_m > 1000 & Elev_m < 2000, "present", "absent"))
```

Now add the predicted presence using a `geom_tile()` layer to make an overlay (filtering to just the presences so absences are blank).

```R
ggplot() +
    geom_raster(data=swissdem,
                aes(x=x, y=y, fill=Elev_m)) +
    scale_fill_gradientn(colors=terrain.colors(22), name="Elevation (m)") +
    geom_tile(data=filter(swissdem, present=="present"), 
              aes(x=x, y=y), fill="blue", 
              alpha=0.6) +
    coord_quickmap() +
    labs(title="Predicted distribution of Willow Tit in Switzerland") +
    theme_void() +
    theme(plot.title=element_text(hjust=0.5, vjust=-2))
```

 A bit more finessing using `scale_fill_manual()` and the package `ggnewscale` gives us separate legends and color scales.

```r
library(ggnewscale)

ggplot() +
    geom_raster(data=swissdem,
                aes(x=x, y=y, fill=Elev_m)) +
    scale_fill_gradientn(colors=terrain.colors(22), name="Elevation (m)") +
    new_scale_fill() +
    geom_tile(data=filter(swissdem, present=="present"), 
              aes(x=x, y=y, fill="Present"), 
              alpha=0.6) +
    scale_fill_manual(name="Willow Tit", 
                      breaks=c("Present"), 
                      values=c("Present"="blue")) +
    coord_quickmap() +
    labs(title="Predicted distribution of Willow Tit in Switzerland") +
    theme_void() +
    theme(plot.title=element_text(hjust=0.5, vjust=-2))
```

You might try other color schemes. For example, `scale_fill_viridis_c()` with a pink overlay looks pretty good to me too.

**Push your code to GitHub**



#### 2. Reading for Monday: decision tree algorithms

James et al. Chapter 8.1: The Basics of Decision Trees

#### 3. Reading for Wednesday
James et al. Chapter 8.2: Bagging, Random Forests and Boosting.
