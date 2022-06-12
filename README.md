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


