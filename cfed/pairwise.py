import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances as ground_euclidean_distances
from scipy.sparse import coo_matrix


def __validate_dataframes(df1, df2):
    assert type(df1) == type(df2) == type(pd.DataFrame()), 'DataFrames must be passed'
    assert df1.shape[1] == df2.shape[1], 'Number of columns is not the same'
    assert all(col in df2.columns for col in df1.columns) and all(col in df1.columns for col in df2.columns),\
        'Columns do not match'
    assert all(df1.dtypes[col] == df2.dtypes[col] for col in df1.columns), 'Columns data types do not match'


def __get_dummies(df1_categorical, df2_categorical, categorical_features=None):
    df1_length = df1_categorical.shape[0]

    df_tmp = pd.concat([df1_categorical, df2_categorical])
    df_tmp = pd.get_dummies(df_tmp, columns=categorical_features)

    df1_categorical = coo_matrix(df_tmp[:df1_length].values)
    df2_categorical = coo_matrix(df_tmp[df1_length:].values)
    return df1_categorical, df2_categorical


def euclidean_distances(df1, df2, categorical_columns=None):
    __validate_dataframes(df1, df2)

    if categorical_columns is not None:
        for col in categorical_columns:
            df1 = df1.astype({col: 'category'})
            df2 = df2.astype({col: 'category'})

    numerical_features = []
    categorical_features = []

    for col in df1.columns:
        if df1[col].dtype.name == 'object' or df1[col].dtype.name == 'category':
            categorical_features.append(col)
        else:
            numerical_features.append(col)

    categorical_features_count = len(categorical_features)
    categorical_features_count_squared = categorical_features_count**2

    df1_numerical = df1[numerical_features]
    df1_categorical = df1[categorical_features]
    df2_numerical = df2[numerical_features]
    df2_categorical = df2[categorical_features]

    distances = np.power(ground_euclidean_distances(df1_numerical, df2_numerical), 2)

    if df1_categorical.shape[0] > 0 and df1_categorical.shape[1] > 0:
        df1_categorical, df2_categorical = __get_dummies(df1_categorical, df2_categorical,
                                                         categorical_features=categorical_features)
        matched_features = df1_categorical.dot(df2_categorical.T).toarray()
        categorical_distances = 2 * (categorical_features_count_squared - matched_features)
        distances += categorical_distances

    distances = np.sqrt(distances)

    return distances
