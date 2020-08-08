from cfed.pairwise import euclidean_distances
from cfed.pairwise import euclidean_distances_from_split
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


def test_split_basic_functionality():
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
        'col4': [1, 1, 2],
    })

    df2_categorical = pd.DataFrame.from_dict({
        'col4': [1, 2, 2],
    })

    distances = euclidean_distances_from_split(df1_numerical, df1_categorical, df2_numerical, df2_categorical)
    assert distances.shape == (df1_numerical.shape[0], df2_numerical.shape[0])
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


def __test_split_empty_dataframes():
    try:
        df1_numerical = pd.DataFrame()
        df2_numerical = pd.DataFrame.from_dict({
            'col1': [1, 4, 7],
            'col2': [2, 5, 8],
            'col3': [3, 6, 9],
        })

        df1_categorical = pd.DataFrame()
        df2_categorical = pd.DataFrame.from_dict({
            'col4': [1, 2, 2],
            'col5': [5, 6, 5],
        })

        euclidean_distances_from_split(df1_numerical, df1_categorical, df2_numerical, df2_categorical)
        assert False, 'DataFrame validation failed'
    except AssertionError as e:
        pass

    try:
        df1_numerical = pd.DataFrame.from_dict({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col3': [7, 8, 9],
            'col5': [9, 1, 6],
        })
        df2_numerical = pd.DataFrame()

        df1_categorical = pd.DataFrame.from_dict({
            'col4': [1, 1, 2],
        })
        df2_categorical = pd.DataFrame()

        euclidean_distances_from_split(df1_numerical, df1_categorical, df2_numerical, df2_categorical)
        assert False, 'DataFrame validation failed'
    except AssertionError as e:
        pass


def __test_split_different_length():
    try:
        df1_numerical = pd.DataFrame.from_dict({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col3': [7, 8, 9],
        })
        df2_numerical = pd.DataFrame.from_dict({
            'col1': [1, 4, 7, 8],
            'col2': [2, 5, 8, 9],
            'col3': [3, 6, 9, 0],
        })

        df1_categorical = pd.DataFrame.from_dict({
            'col4': [1, 1, 2],
        })

        df2_categorical = pd.DataFrame.from_dict({
            'col4': [1, 2, 2, 3],
        })

        euclidean_distances_from_split(df1_numerical, df1_categorical, df2_numerical, df2_categorical)
        assert False, 'DataFrame validation failed'
    except AssertionError as e:
        pass


def __test_split_different_column_count():
    try:
        df1_numerical = pd.DataFrame.from_dict({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col3': [7, 8, 9],
        })
        df2_numerical = pd.DataFrame.from_dict({
            'col1': [1, 4, 7],
            'col2': [2, 5, 8],
            'col3': [3, 6, 9],
        })

        df1_categorical = pd.DataFrame.from_dict({
            'col4': [1, 1, 2],
        })

        df2_categorical = pd.DataFrame.from_dict({
            'col4': [1, 2, 2],
            'col5': [5, 6, 5],
        })

        euclidean_distances_from_split(df1_numerical, df1_categorical, df2_numerical, df2_categorical)
        assert False, 'DataFrame validation failed'
    except AssertionError as e:
        pass

    try:
        df1_numerical = pd.DataFrame.from_dict({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col3': [7, 8, 9],
            'col5': [9, 1, 6],
        })
        df2_numerical = pd.DataFrame.from_dict({
            'col1': [1, 4, 7],
            'col2': [2, 5, 8],
            'col3': [3, 6, 9],
        })

        df1_categorical = pd.DataFrame.from_dict({
            'col4': [1, 1, 2],
        })

        df2_categorical = pd.DataFrame.from_dict({
            'col4': [1, 2, 2],
        })

        euclidean_distances_from_split(df1_numerical, df1_categorical, df2_numerical, df2_categorical)
        assert False, 'DataFrame validation failed'
    except AssertionError as e:
        pass


def __test_split_different_column_names():
    try:
        df1_numerical = pd.DataFrame.from_dict({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col4': [7, 8, 9],
        })
        df2_numerical = pd.DataFrame.from_dict({
            'col1': [1, 4, 7],
            'col2': [2, 5, 8],
            'col3': [3, 6, 9],
        })

        df1_categorical = pd.DataFrame.from_dict({
            'col4': [1, 1, 2],
        })

        df2_categorical = pd.DataFrame.from_dict({
            'col4': [1, 2, 2],
        })

        euclidean_distances_from_split(df1_numerical, df1_categorical, df2_numerical, df2_categorical)
        assert False, 'DataFrame validation failed'
    except AssertionError as e:
        pass

    try:
        df1_numerical = pd.DataFrame.from_dict({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col3': [7, 8, 9],
        })
        df2_numerical = pd.DataFrame.from_dict({
            'col1': [1, 4, 7],
            'col2': [2, 5, 8],
            'col3': [3, 6, 9],
        })

        df1_categorical = pd.DataFrame.from_dict({
            'col4': [1, 1, 2],
        })

        df2_categorical = pd.DataFrame.from_dict({
            'col5': [1, 2, 2],
        })

        euclidean_distances_from_split(df1_numerical, df1_categorical, df2_numerical, df2_categorical)
        assert False, 'DataFrame validation failed'
    except AssertionError as e:
        pass


def __test_split_different_column_dtypes():
    try:
        df1_numerical = pd.DataFrame.from_dict({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col3': [7, 8, '9'],
        })

        df2_numerical = pd.DataFrame.from_dict({
            'col1': [1, 4, 7],
            'col2': [2, 5, 8],
            'col3': [3, 6, 9],
        })

        df1_categorical = pd.DataFrame.from_dict({
            'col4': [1, 1, 2],
        })

        df2_categorical = pd.DataFrame.from_dict({
            'col4': [1, 2, 2],
        })

        euclidean_distances_from_split(df1_numerical, df1_categorical, df2_numerical, df2_categorical)
        assert False, 'DataFrame validation failed'
    except AssertionError as e:
        pass

    try:
        df1_numerical = pd.DataFrame.from_dict({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col3': [7, 8, 9],
        })
        df2_numerical = pd.DataFrame.from_dict({
            'col1': [1, 4, 7],
            'col2': [2, 5, 8],
            'col3': [3, 6, 9],
        })

        df1_categorical = pd.DataFrame.from_dict({
            'col4': [1, 1, 2],
        })

        df2_categorical = pd.DataFrame.from_dict({
            'col4': [1, 2, '2'],
        })

        euclidean_distances_from_split(df1_numerical, df1_categorical, df2_numerical, df2_categorical)
        assert False, 'DataFrame validation failed'
    except AssertionError as e:
        pass


def test_split_dataframes_validation():
    __test_split_empty_dataframes()
    __test_split_different_length()
    __test_split_different_column_count()
    __test_split_different_column_names()
    __test_split_different_column_dtypes()


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


test_split_dataframes_validation()
