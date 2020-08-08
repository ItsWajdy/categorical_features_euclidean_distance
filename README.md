# Categorical Features Pairwise Euclidean Distances

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)

[![PyPi Version](https://img.shields.io/pypi/v/cfed.svg)](https://pypi.python.org/pypi/cfed/)

A python package to compute pairwise Euclidean distances on datasets with categorical features in little time


## Motivation

In machine learning model development I often ran into datasets with categorical features. Most times dealing with these categorical features was fairly straight forward (I would use the pandas get_dummies() function to convert each feature into a one-hot-encoded representaion).

But when the number of categories embedded in these categorical features became massive, I ran into the problem of extremely slow Euclidean distance computation between each sample and every other sample.

This is where this package comes in. Running my own tests, I concluded that this code runs significantly faster than the [SKLearn pairwise Euclidean distances function](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html) on one-hot-encoded categorical features.


## Prerequisites

See [requirements.txt](https://github.com/ItsWajdy/categorical_features_euclidean_distance/blob/master/requirements.txt) for the full list of Prerequisite libraries.


## Installation

To start using this package, simply run this command in terminal

`pip install cfed`


## Usage

```
import pandas as pd
from cfed.pairwise import euclidean_distances
from cfed.pairwise import euclidean_distances_from_split

df1 = pd.DataFrame.from_dict({
    'col1': [1, 2, 3],
    'col2': [4, 5, 6],
    'col3': [7, 8, 9]
})
df2 = pd.DataFrame.from_dict({
    'col1': [1, 4, 7],
    'col2': [2, 5, 8],
    'col3': [3, 6, 9],
})

distances = euclidean_distances(df1, df2, categorical_columns=['col1'])
```

*Or without specifying `categorical_columns`*

```
import pandas as pd
from cfed.pairwise import euclidean_distances
from cfed.pairwise import euclidean_distances_from_split

df1 = pd.DataFrame.from_dict({
    'col1': ['c1', 'c2', 'c1'],
    'col2': [4, 5, 6],
    'col3': [7, 8, 9]
})
df2 = pd.DataFrame.from_dict({
    'col1': ['c1', 'c3', 'c2'],
    'col2': [2, 5, 8],
    'col3': [3, 6, 9],
})

distances = euclidean_distances(df1, df2)
```

*Or*

```
import pandas as pd
from cfed.pairwise import euclidean_distances
from cfed.pairwise import euclidean_distances_from_split

df1_numerical = pd.DataFrame.from_dict({
    'col1': [1, 2, 3],
    'col2': [4, 5, 6],
    'col3': [7, 8, 9]
})
df2_numerical = pd.DataFrame.from_dict({
    'col1': [1, 4, 7],
    'col2': [2, 5, 8],
    'col3': [3, 6, 9],
})

df1_categorical = pd.DataFrame.from_dict({
    'col4': ['c1', 'c1', 'c2'],
})
df2_categorical = pd.DataFrame.from_dict({
    'col4': ['c1', 'c2', 'c2'],
})

distances = euclidean_distances_from_split(df1_numerical, df1_categorical, df2_numerical, df2_categorical)
```
