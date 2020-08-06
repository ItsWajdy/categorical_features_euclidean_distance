from cfed.pairwise import euclidean_distances
import pandas as pd
import numpy as np
import sklearn.metrics.pairwise as pw


def test_basic_functionality():
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
    assert distances.shape == (df1.shape[0], df2.shape[0])
    assert not any(elem == np.nan for row in distances for elem in row)


def __test_different_column_count():
    try:
        df1 = pd.DataFrame.from_dict({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col3': [7, 8, 9]
        })
        df2 = pd.DataFrame.from_dict({
            'col1': [1, 4, 7],
            'col2': [2, 5, 8],
            'col4': [3, 6, 9],
            'col5': [10, 10, 10],
        })

        euclidean_distances(df1, df2, categorical_columns=['col1'])
        assert False, 'DataFrame validation failed'
    except AssertionError as e:
        pass


def __test_different_column_names():
    try:
        df1 = pd.DataFrame.from_dict({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col3': [7, 8, 9]
        })
        df2 = pd.DataFrame.from_dict({
            'col1': [1, 4, 7],
            'col2': [2, 5, 8],
            'col4': [3, 6, 9],
        })

        euclidean_distances(df1, df2, categorical_columns=['col1'])
        assert False, 'DataFrame validation failed'
    except AssertionError as e:
        pass


def __test_different_column_dtypes():
    try:
        df1 = pd.DataFrame.from_dict({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col3': [7, 8, 9]
        })
        df2 = pd.DataFrame.from_dict({
            'col1': [1, 4, 7],
            'col2': [2, 5, 8],
            'col4': [3, 6, '9'],
        })

        euclidean_distances(df1, df2, categorical_columns=['col1'])
        assert False, 'DataFrame validation failed'
    except AssertionError as e:
        pass


def test_dataframes_validation():
    __test_different_column_count()
    __test_different_column_names()
    __test_different_column_dtypes()


def test_equality_no_categorical_features():
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

    predicted_truth = euclidean_distances(df1, df2)
    ground_truth = pw.euclidean_distances(df1.values, df2.values)

    assert np.array_equal(predicted_truth, ground_truth),\
        'Results differ from ground-truth Euclidean distances in case no categorical features'


def test_non_equality_with_categorical_features():
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

    predicted_truth = euclidean_distances(df1, df2, categorical_columns=['col1'])
    ground_truth = pw.euclidean_distances(df1.values, df2.values)

    assert not np.array_equal(predicted_truth, ground_truth)
