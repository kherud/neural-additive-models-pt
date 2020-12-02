# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Data readers for regression/ binary classification datasets."""

import os
import glob
import gzip
import os.path as osp
import tarfile
from typing import Tuple, Dict, Union, Iterator, List

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = 'gs://nam_datasets/data'
DatasetType = Tuple[np.ndarray, np.ndarray]


def save_array_to_disk(filename,
                       np_arr,
                       allow_pickle=False):
    """Saves a np.ndarray to a specified file on disk."""
    np.save(filename, np_arr, allow_pickle=allow_pickle)


def read_dataset(dataset_name,
                 header='infer',
                 names=None,
                 delim_whitespace=False):
    dataset_path = osp.join(DATA_PATH, dataset_name)
    return pd.read_csv(dataset_path, header=header, names=names, delim_whitespace=delim_whitespace)


def load_breast_data():
    """Load and return the Breast Cancer Wisconsin dataset (classification)."""

    breast = load_breast_cancer()
    feature_names = list(breast.feature_names)
    return {
        'problem': 'classification',
        'X': pd.DataFrame(breast.data, columns=feature_names),
        'y': breast.target,
    }


def load_adult_data():
    """Loads the Adult Income dataset.

    Predict whether income exceeds $50K/yr based on census data. Also known as
    "Census Income" dataset. For more info, see
    https://archive.ics.uci.edu/ml/datasets/Adult.

    Returns:
      A dict containing the `problem` type (regression or classification) and the
      input features `X` as a pandas.Dataframe and the labels `y` as a pd.Series.
    """
    df = read_dataset('adult.data', header=None)
    df.columns = [
        'Age', 'WorkClass', 'fnlwgt', 'Education', 'EducationNum',
        'MaritalStatus', 'Occupation', 'Relationship', 'Race', 'Gender',
        'CapitalGain', 'CapitalLoss', 'HoursPerWeek', 'NativeCountry', 'Income'
    ]
    train_cols = df.columns[0:-1]
    label = df.columns[-1]
    x_df = df[train_cols]
    y_df = df[label]
    return {'problem': 'classification', 'X': x_df, 'y': y_df}


def load_heart_data():
    """Loads the Heart Disease dataset.

    The Cleveland Heart Disease Data found in the UCI machine learning repository
    consists of 14 variables measured on 303 individuals who have heart disease.
    See https://www.kaggle.com/sonumj/heart-disease-dataset-from-uci for more
    info.

    Returns:
      A dict containing the `problem` type (regression or classification) and the
      input features `X` as a pandas.Dataframe and the labels `y` as a pd.Series.
    """
    df = read_dataset('HeartDisease.csv')
    train_cols = df.columns[0:-2]
    label = df.columns[-2]
    x_df = df[train_cols]
    y_df = df[label]
    # Replace NaN values with the mode value in the column.
    for col_name in x_df.columns:
        x_df[col_name].fillna(x_df[col_name].mode()[0], inplace=True)
    return {
        'problem': 'classification',
        'X': x_df,
        'y': y_df,
    }


def load_credit_data():
    """Loads the Credit Fraud Detection dataset.

    This dataset contains transactions made by credit cards in September 2013 by
    european cardholders. It presents transactions that occurred in 2 days, where
    we have 492 frauds out of 284,807 transactions. It is highly unbalanced, the
    positive class (frauds) account for 0.172% of all transactions.
    See https://www.kaggle.com/mlg-ulb/creditcardfraud for more info.

    Returns:
      A dict containing the `problem` type (i.e. classification) and the
      input features `X` as a pandas.Dataframe and the labels `y` as a pd.Series.
    """
    df = read_dataset('creditcard.csv')
    df = df.dropna()
    train_cols = df.columns[0:-1]
    label = df.columns[-1]
    x_df = df[train_cols]
    y_df = df[label]
    return {
        'problem': 'classification',
        'X': x_df,
        'y': y_df,
    }

if __name__ == "__main__":
    load_credit_data()


def load_telco_churn_data():
    """Loads Telco Customer Churn dataset.

    Predict behavior to retain customers based on relevant customer data.
    See https://www.kaggle.com/blastchar/telco-customer-churn/ for more info.

    Returns:
      A dict containing the `problem` type (i.e. classification) and the
      input features `X` as a pandas.Dataframe and the labels `y` as a pd.Series.
    """
    df = read_dataset('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    train_cols = df.columns[1:-1]  # First column is an ID
    label = df.columns[-1]
    x_df = df[train_cols]
    # Impute missing values
    x_df['TotalCharges'] = x_df['TotalCharges'].replace(' ', 0).astype('float64')
    y_df = df[label]  # 'Yes', 'No'.
    return {
        'problem': 'classification',
        'X': x_df,
        'y': y_df,
    }


class CustomPipeline(Pipeline):
    """Custom sklearn Pipeline to transform data."""

    def apply_transformation(self, x):
        """Applies all transforms to the data, without applying last estimator.

        Args:
          x: Iterable data to predict on. Must fulfill input requirements of first
            step of the pipeline.

        Returns:
          xt: Transformed data.
        """
        xt = x
        for _, transform in self.steps[:-1]:
            xt = transform.fit_transform(xt)
        return xt


def transform_data(df):
    """Apply a fixed set of transformations to the pd.Dataframe `df`.

    Args:
      df: Input dataframe containing features.

    Returns:
      Transformed dataframe and corresponding column names. The transformations
      include (1) encoding categorical features as a one-hot numeric array, (2)
      identity `FunctionTransformer` for numerical variables. This is followed by
      scaling all features to the range (-1, 1) using min-max scaling.
    """
    column_names = df.columns
    new_column_names = []
    is_categorical = np.array([dt.kind == 'O' for dt in df.dtypes])
    categorical_cols = df.columns.values[is_categorical]
    numerical_cols = df.columns.values[~is_categorical]
    for index, is_cat in enumerate(is_categorical):
        col_name = column_names[index]
        if is_cat:
            new_column_names += [
                '{}: {}'.format(col_name, val) for val in set(df[col_name])
            ]
        else:
            new_column_names.append(col_name)
    cat_ohe_step = ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))

    cat_pipe = Pipeline([cat_ohe_step])
    num_pipe = Pipeline([('identity', FunctionTransformer(validate=True))])
    transformers = [('cat', cat_pipe, categorical_cols),
                    ('num', num_pipe, numerical_cols)]
    column_transform = ColumnTransformer(transformers=transformers)

    pipe = CustomPipeline([('column_transform', column_transform),
                           ('min_max', MinMaxScaler((-1, 1))), ('dummy', None)])
    df = pipe.apply_transformation(df)
    return df, new_column_names


def load_dataset(dataset_name):
    """Loads the dataset according to the `dataset_name` passed.

    Args:
      dataset_name: Name of the dataset to be loaded.

    Returns:
      data_x: np.ndarray of size (n_examples, n_features) containining the
        features per input data point where n_examples is the number of examples
        and n_features is the number of features.
      data_y: np.ndarray of size (n_examples, ) containing the label/target
        for each example where n_examples is the number of examples.
      column_names: A list containing the feature names.

    Raises:
      ValueError: If the `dataset_name` is not in ('Telco', 'BreastCancer',
      'Adult', 'Credit', 'Heart', 'Mimic2', 'Recidivism', 'Fico', Housing').
    """
    if dataset_name == 'Telco':
        dataset = load_telco_churn_data()
    elif dataset_name == 'BreastCancer':
        dataset = load_breast_data()
    elif dataset_name == 'Adult':
        dataset = load_adult_data()
    elif dataset_name == 'Credit':
        dataset = load_credit_data()
    elif dataset_name == 'Heart':
        dataset = load_heart_data()
    else:
        raise ValueError('{} not found!'.format(dataset_name))

    data_x, data_y = dataset['X'].copy(), dataset['y'].copy()
    problem_type = dataset['problem']
    data_x, column_names = transform_data(data_x)
    data_x = data_x.astype('float32')
    if (problem_type == 'classification') and \
            (not isinstance(data_y, np.ndarray)):
        data_y = pd.get_dummies(data_y).values
        data_y = np.argmax(data_y, axis=-1)
    data_y = data_y.astype('float32')
    return data_x, data_y, column_names


def get_train_test_fold(
        data_x,
        data_y,
        fold_num,
        num_folds,
        stratified=True,
        random_state=42):
    """Returns a specific fold split for K-Fold cross validation.

    Randomly split dataset into `num_folds` consecutive folds and returns the fold
    with index `fold_index` for testing while the `num_folds` - 1 remaining folds
    form the training set.

    Args:
      data_x: Training data, with shape (n_samples, n_features), where n_samples
        is the number of samples and n_features is the number of features.
      data_y: The target variable, with shape (n_samples), for supervised learning
        problems.  Stratification is done based on the y labels.
      fold_num: Index of fold used for testing.
      num_folds: Number of folds.
      stratified: Whether to preserve the percentage of samples for each class in
        the different folds (only applicable for classification).
      random_state: Seed used by the random number generator.

    Returns:
      (x_train, y_train): Training folds containing 1 - (1/`num_folds`) fraction
        of entire data.
      (x_test, y_test): Test fold containing 1/`num_folds` fraction of data.
    """
    if stratified:
        stratified_k_fold = StratifiedKFold(
            n_splits=num_folds, shuffle=True, random_state=random_state)
    else:
        stratified_k_fold = KFold(
            n_splits=num_folds, shuffle=True, random_state=random_state)
    assert fold_num <= num_folds and fold_num > 0, 'Pass a valid fold number.'
    for train_index, test_index in stratified_k_fold.split(data_x, data_y):
        if fold_num == 1:
            x_train, x_test = data_x[train_index], data_x[test_index]
            y_train, y_test = data_y[train_index], data_y[test_index]
            return (x_train, y_train), (x_test, y_test)
        else:
            fold_num -= 1


def split_training_dataset(
        data_x,
        data_y,
        n_splits,
        stratified=True,
        test_size=0.125,
        random_state=1337):
    """Yields a generator that randomly splits data into (train, validation) set.

    The train set is used for fitting the DNNs/NAMs while the validation set is
    used for early stopping.

    Args:
      data_x: Training data, with shape (n_samples, n_features), where n_samples
        is the number of samples and n_features is the number of features.
      data_y: The target variable, with shape (n_samples), for supervised learning
        problems.  Stratification is done based on the y labels.
      n_splits: Number of re-shuffling & splitting iterations.
      stratified: Whether to preserve the percentage of samples for each class in
        the (train, validation) splits. (only applicable for classification).
      test_size: The proportion of the dataset to include in the validation split.
      random_state: Seed used by the random number generator.

    Yields:
      (x_train, y_train): The training data split.
      (x_validation, y_validation): The validation data split.
    """
    if stratified:
        stratified_shuffle_split = StratifiedShuffleSplit(
            n_splits=n_splits, test_size=test_size, random_state=random_state)
    else:
        stratified_shuffle_split = ShuffleSplit(
            n_splits=n_splits, test_size=test_size, random_state=random_state)
    split_gen = stratified_shuffle_split.split(data_x, data_y)

    for train_index, validation_index in split_gen:
        x_train, x_validation = data_x[train_index], data_x[validation_index]
        y_train, y_validation = data_y[train_index], data_y[validation_index]
        assert x_train.shape[0] == y_train.shape[0]
        yield (x_train, y_train), (x_validation, y_validation)


def create_test_train_fold(
        dataset: Union[str, Tuple[pd.DataFrame, pd.DataFrame]],
        id_fold: int,
        n_folds: int,
        n_splits: int,
        regression: bool = False,
):
    """Splits the dataset into training and held-out test set."""
    data_x, data_y, _ = load_dataset(dataset)
    # Get the training and test set based on the StratifiedKFold split
    (x_train_all, y_train_all), test_dataset = get_train_test_fold(
        data_x,
        data_y,
        fold_num=id_fold,
        num_folds=n_folds,
        stratified=regression)
    data_gen = split_training_dataset(
        x_train_all,
        y_train_all,
        n_splits,
        stratified=regression)
    return data_gen, test_dataset


def calculate_n_units(x_train, n_basis_functions, units_multiplier):
    num_unique_vals = [
        len(np.unique(x_train[:, i])) for i in range(x_train.shape[1])
    ]
    return [
        min(n_basis_functions, i * units_multiplier) for i in num_unique_vals
    ]


def load_pm_data_without_distances(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    df = None
    for file_path in glob.glob(os.path.join(data_dir, "*200.csv")):
        if df is not None:
            df = df.append(pd.read_csv(file_path))
        else:
            df = pd.read_csv(file_path)

    data_x, data_y = df.loc[:, df.columns != "pm_measurement"], df.loc[:, df.columns == "pm_measurement"]
    data_x, column_names = transform_data(data_x)
    data_y, _ = transform_data(data_y)
    data_x = data_x.astype('float32')
    data_y = data_y.astype('float32')
    return data_x, data_y, column_names

def load_pm_data_with_distances(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    cols = ['latitude', 'longitude', 'target']
    for feat in ['commercial', 'industrial', 'residential']:
        for dist in range(50, 3050, 50):
            cols.append(f"{feat}_{dist}")
    for feat in ['bigstreet', 'localstreet']:
        for dist in range(50, 1550, 50):
            cols.append(f"{feat}_{dist}")
    cols.append('dist_trafficsignal')
    cols.append('dist_motorway')
    cols.append('dist_primaryroad')
    cols.append('dist_industrial')

    df = None
    for file_path in glob.glob(os.path.join(data_dir, "*withDistances.csv")):
        if df is not None:
            df = df.append(pd.read_csv(file_path, names=cols))
        else:
            df = pd.read_csv(file_path, names=cols)

    data_x, data_y = df.loc[:, df.columns != "target"], df.loc[:, df.columns == "target"]
    data_x, column_names = transform_data(data_x)
    data_y, _ = transform_data(data_y)
    data_x = data_x.astype('float32')
    data_y = data_y.astype('float32')
    return data_x, data_y, column_names