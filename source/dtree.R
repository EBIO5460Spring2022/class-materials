# Fit a tree with d splits
# It's strange there is not a more efficient way to do this using `tree()` but
# apparently there is not a stopping rule for number of splits. This is a hack
# that is very slow because we first fit the full tree, then snip off the
# branches above the number of desired splits. It would of course be much faster
# to build a tree, stopping after d splits.
#
# formula:    model formula (formula)
# data:       y and x data (data.frame, mixed)
# d:          number of splits (scalar, integer)
# return:     bagged predictions (tree)
#
dtree <- function(formula, data, d) {
    
#   Fit maximum tree
    maxtree_pars <- tree.control(nobs=nrow(data), mindev=0, minsize=2)
    maxtree <- tree(formula, data, control=maxtree_pars)

#   Get info from max tree
    nodes <- as.integer(rownames(maxtree$frame))
    node_deviance <- maxtree$frame$dev
    non_leaf_indicator <- maxtree$frame$var != "<leaf>"
    non_leaf_nodes <- nodes[non_leaf_indicator]
    non_leaf_deviance <- node_deviance[non_leaf_indicator]
    
#   Sort nodes by decreasing deviance
    non_leaf_nodes <- non_leaf_nodes[order(non_leaf_deviance, decreasing=TRUE)]
    
#   Snip the nodes (least deviance first)
    n_nodes <- length(non_leaf_nodes) #= no. of splits in maxtree
    if ( n_nodes < d ) stop("d exceeds n_nodes; set d < nrow(data)")
    if ( n_nodes == d ) { #case for d = nrow(data) - 1
        dtree <- maxtree
        warning("n_nodes = d") #should only get here on purpose (i.e. d=nrow(data)-1)
    } else {
        nodes2snip <- non_leaf_nodes[(d+1):n_nodes] #all nodes above d
        dtree <- snip.tree(maxtree, nodes2snip)
    }
    
    return(dtree)
}
