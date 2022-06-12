# Robust Random Cut Forest
An implementation of the "Robust Random Cut Forest" algorithm for anomaly detection in streams.

##  What is "Robust Random Cut Forest"?
The **Robust Random Cut Forest (RRCF) algorithm** is an ensemble method for detecting outliers in streaming data. RRCF offers a number of features that many competing anomaly detection algorithms lack. Specifically, RRCF:
- Is designed to handle streaming data.
- Performs well on high-dimensional data.
- Reduces the influence of irrelevant dimensions.
- Gracefully handles duplicates and near-duplicates that could otherwise mask the presence of outliers.
- Features an anomaly-scoring algorithm with a clear underlying statistical meaning.

This repository provides an open-source implementation of the RRCF algorithm and its core data structures for the purposes of facilitating experimentation and enabling future extensions of the RRCF algorithm.

## Installation
Use pip to install rrcf:
```ruby
$ pip install rrcf
```

## Robut Random Cut Trees
A Robust Random Cut Tree (RRCT) is a binary search tree that can be used to detect outliers in a point set. A RRCT can be instantiated from a point set. Points can also be added and removed from an RRCT.

### Creating the Tree
```ruby
import numpy as np
import rrcf

# A (robust) random cut tree can be instantiated from a point set (n x d)
X = np.random.randn(100, 2)
tree = rrcf.RCTree(X)

# A random cut tree can also be instantiated with no points
tree = rrcf.RCTree()
```

### Inserting Points
```ruby
tree = rrcf.RCTree()

for i in range(6):
    x = np.random.randn(2)
    tree.insert_point(x, index=i)
```

```ruby
─+
 ├───+
 │   ├───+
 │   │   ├──(0)
 │   │   └───+
 │   │       ├──(5)
 │   │       └──(4)
 │   └───+
 │       ├──(2)
 │       └──(3)
 └──(1)
```

### Deleting Points
```ruby
tree.forget_point(2)
```

```ruby
─+
 ├───+
 │   ├───+
 │   │   ├──(0)
 │   │   └───+
 │   │       ├──(5)
 │   │       └──(4)
 │   └──(3)
 └──(1)
 ```
 
 ## Anomaly Score
 The likelihood that a point is an outlier is measured by its collusive displacement (CoDisp): if including a new point significantly changes the model complexity (i.e. bit depth), then that point is more likely to be an outlier.

```ruby
# Seed tree with zero-mean, normally distributed data
X = np.random.randn(100,2)
tree = rrcf.RCTree(X)

# Generate an inlier and outlier point
inlier = np.array([0, 0])
outlier = np.array([4, 4])

# Insert into tree
tree.insert_point(inlier, index='inlier')
tree.insert_point(outlier, index='outlier')
```

```ruby
tree.codisp('inlier')
>>> 1.75
```

```ruby
tree.codisp('outlier')
>>> 39.0
```
