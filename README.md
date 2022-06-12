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

## Batch Anomaly Detection
This example shows how a robust random cut forest can be used to detect outliers in a batch setting. Outliers correspond to large CoDisp.

```ruby
import numpy as np
import pandas as pd
import rrcf

# Set parameters
np.random.seed(0)
n = 2010
d = 3
num_trees = 100
tree_size = 256

# Generate data
X = np.zeros((n, d))
X[:1000,0] = 5
X[1000:2000,0] = -5
X += 0.01*np.random.randn(*X.shape)

# Construct forest
forest = []
while len(forest) < num_trees:
    # Select random subsets of points uniformly from point set
    ixs = np.random.choice(n, size=(n // tree_size, tree_size),
                           replace=False)
    # Add sampled trees to forest
    trees = [rrcf.RCTree(X[ix], index_labels=ix) for ix in ixs]
    forest.extend(trees)

# Compute average CoDisp
avg_codisp = pd.Series(0.0, index=np.arange(n))
index = np.zeros(n)
for tree in forest:
    codisp = pd.Series({leaf : tree.codisp(leaf) for leaf in tree.leaves})
    avg_codisp[codisp.index] += codisp
    np.add.at(index, codisp.index.values, 1)
avg_codisp /= index
```

![Image not found](https://github.com/aryaninamdar/Robust-Random-Cut-Forest/blob/main/examples/example1-1.png)

## Streaming Anomaly Detection
This example shows how the algorithm can be used to detect anomalies in streaming time series data.

```ruby
import numpy as np
import rrcf

# Generate data
n = 730
A = 50
center = 100
phi = 30
T = 2*np.pi/100
t = np.arange(n)
sin = A*np.sin(T*t-phi*T) + center
sin[235:255] = 80

# Set tree parameters
num_trees = 40
shingle_size = 4
tree_size = 256

# Create a forest of empty trees
forest = []
for _ in range(num_trees):
    tree = rrcf.RCTree()
    forest.append(tree)
    
# Use the "shingle" generator to create rolling window
points = rrcf.shingle(sin, size=shingle_size)

# Create a dict to store anomaly score of each point
avg_codisp = {}

# For each shingle...
for index, point in enumerate(points):
    # For each tree in the forest...
    for tree in forest:
        # If tree is above permitted size, drop the oldest point (FIFO)
        if len(tree.leaves) > tree_size:
            tree.forget_point(index - tree_size)
        # Insert the new point into the tree
        tree.insert_point(point, index=index)
        # Compute codisp on the new point and take the average among all trees
        if not index in avg_codisp:
            avg_codisp[index] = 0
        avg_codisp[index] += tree.codisp(index) / num_trees
```ruby

![Image not found](https://github.com/aryaninamdar/Robust-Random-Cut-Forest/blob/main/examples/example2-2.png)
