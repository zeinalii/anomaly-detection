import warnings
import os
import json
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler


warnings.filterwarnings("ignore")
np.random.seed(203)
COLOR_PALETTE = ["#0101DF", "#DF0101"]


def create_balanced_subset(dataframe, debug=False):
    if debug:
        print(f"\nNo Fraud: {round(dataframe['Class'].value_counts()[0] / len(dataframe) * 100, 2)}%")
        print(f"Fraud: {round(dataframe['Class'].value_counts()[1] / len(dataframe) * 100, 2)}%\n")

    scaler_standard = StandardScaler()
    scaler_robust = RobustScaler()

    dataframe['amount_scaled'] = scaler_robust.fit_transform(dataframe['Amount'].values.reshape(-1, 1))
    dataframe['time_scaled'] = scaler_robust.fit_transform(dataframe['Time'].values.reshape(-1, 1))

    dataframe.drop(['Time', 'Amount'], axis=1, inplace=True)

    amount = dataframe.pop('amount_scaled')
    time = dataframe.pop('time_scaled')

    dataframe.insert(0, 'amount_scaled', amount)
    dataframe.insert(1, 'time_scaled', time)

    if debug:
        dataframe.head()

    features = dataframe.drop('Class', axis=1)
    labels = dataframe['Class']

    skf = StratifiedKFold(n_splits=5)
    for train_idx, test_idx in skf.split(features, labels):
        original_X_train, original_X_test = features.iloc[train_idx], features.iloc[test_idx]
        original_y_train, original_y_test = labels.iloc[train_idx], labels.iloc[test_idx]

    original_X_train = original_X_train.values
    original_X_test = original_X_test.values
    original_y_train = original_y_train.values
    original_y_test = original_y_test.values

    if debug:
        print('-' * 100)
        print('Training Label Distribution:', original_y_train.mean())
        print('Testing Label Distribution:', original_y_test.mean())

    shuffled_df = dataframe.sample(frac=1)
    fraud_data = shuffled_df[shuffled_df['Class'] == 1]
    non_fraud_data = shuffled_df[shuffled_df['Class'] == 0].head(len(fraud_data))

    balanced_df = pd.concat([fraud_data, non_fraud_data]).sample(frac=1, random_state=42)

    if debug:
        print("\nBalanced Dataset:")
        print(f"No Fraud: {round(balanced_df['Class'].value_counts()[0] / len(balanced_df) * 100, 2)}%")
        print(f"Fraud: {round(balanced_df['Class'].value_counts()[1] / len(balanced_df) * 100, 2)}%\n")
        print('Class Distribution:', balanced_df['Class'].value_counts() / len(balanced_df))

    return balanced_df


def analyze_correlations(original, subset):
    fig, axes = plt.subplots(2, 1, figsize=(24, 20))
    corr_full = original.corr()
    sns.heatmap(corr_full, cmap='coolwarm_r', annot_kws={'size': 20}, ax=axes[0])
    axes[0].set_title("Full Dataset Correlation Matrix", fontsize=14)

    corr_subset = subset.corr()
    sns.heatmap(corr_subset, cmap='coolwarm_r', annot_kws={'size': 20}, ax=axes[1])
    axes[1].set_title('Subset Correlation Matrix', fontsize=14)
    plt.show()

    fig, axes = plt.subplots(ncols=4, figsize=(20, 4))
    sns.boxplot(x="Class", y="V17", data=subset, palette=COLOR_PALETTE, ax=axes[0])
    axes[0].set_title('V17 vs Class')

    sns.boxplot(x="Class", y="V14", data=subset, palette=COLOR_PALETTE, ax=axes[1])
    axes[1].set_title('V14 vs Class')

    sns.boxplot(x="Class", y="V12", data=subset, palette=COLOR_PALETTE, ax=axes[2])
    axes[2].set_title('V12 vs Class')

    sns.boxplot(x="Class", y="V10", data=subset, palette=COLOR_PALETTE, ax=axes[3])
    axes[3].set_title('V10 vs Class')

    plt.show()

    fig, axes = plt.subplots(ncols=4, figsize=(20, 4))
    sns.boxplot(x="Class", y="V11", data=subset, palette=COLOR_PALETTE, ax=axes[0])
    axes[0].set_title('V11 vs Class')

    sns.boxplot(x="Class", y="V4", data=subset, palette=COLOR_PALETTE, ax=axes[1])
    axes[1].set_title('V4 vs Class')

    sns.boxplot(x="Class", y="V2", data=subset, palette=COLOR_PALETTE, ax=axes[2])
    axes[2].set_title('V2 vs Class')

    sns.boxplot(x="Class", y="V19", data=subset, palette=COLOR_PALETTE, ax=axes[3])
    axes[3].set_title('V19 vs Class')

    plt.show()


def remove_outliers(dataframe, debug=False):
    v14_fraud = dataframe.loc[dataframe['Class'] == 1, 'V14'].values
    q25, q75 = np.percentile(v14_fraud, [25, 75])
    iqr = q75 - q25
    cutoff = iqr * 1.5
    lower, upper = q25 - cutoff, q75 + cutoff
    outliers_v14 = [x for x in v14_fraud if x < lower or x > upper]
    dataframe = dataframe.drop(dataframe[(dataframe['V14'] < lower) | (dataframe['V14'] > upper)].index)

    if debug:
        print(f'V14 - Q25: {q25}, Q75: {q75}, IQR: {iqr}, Lower: {lower}, Upper: {upper}')
        print(f'V14 Outliers: {len(outliers_v14)}')
        print('-' * 176)

    v12_fraud = dataframe.loc[dataframe['Class'] == 1, 'V12'].values
    q25, q75 = np.percentile(v12_fraud, [25, 75])
    iqr = q75 - q25
    cutoff = iqr * 1.5
    lower, upper = q25 - cutoff, q75 + cutoff
    outliers_v12 = [x for x in v12_fraud if x < lower or x > upper]
    dataframe = dataframe.drop(dataframe[(dataframe['V12'] < lower) | (dataframe['V12'] > upper)].index)

    if debug:
        print(f'V12 - Lower: {lower}, Upper: {upper}')
        print(f'V12 Outliers: {len(outliers_v12)}')
        print(f'Instances after V12 removal: {len(dataframe)}')
        print('-' * 176)

    v10_fraud = dataframe.loc[dataframe['Class'] == 1, 'V10'].values
    q25, q75 = np.percentile(v10_fraud, [25, 75])
    iqr = q75 - q25
    cutoff = iqr * 1.5
    lower, upper = q25 - cutoff, q75 + cutoff
    outliers_v10 = [x for x in v10_fraud if x < lower or x > upper]
    dataframe = dataframe.drop(dataframe[(dataframe['V10'] < lower) | (dataframe['V10'] > upper)].index)

    if debug:
        print(f'V10 - Lower: {lower}, Upper: {upper}')
        print(f'V10 Outliers: {len(outliers_v10)}')
        print(f'Instances after V10 removal: {len(dataframe)}')
        print('-' * 176)

    return dataframe


def reduce_dimensions(features, method="pca"):
    method = method.lower()
    if method == "pca":
        reducer = PCA(n_components=2, random_state=42)
    elif method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = TruncatedSVD(n_components=2, random_state=42)
    return reducer.fit_transform(features)


def prepare_data(df, debug=False, dim_reduce=False, reduction_method="pca"):
    balanced_df = create_balanced_subset(df)

    if debug:
        analyze_correlations(df, balanced_df)

    cleaned_df = remove_outliers(balanced_df)

    X = cleaned_df.drop('Class', axis=1)
    y = cleaned_df['Class']

    if dim_reduce:
        X = reduce_dimensions(X, reduction_method)

    X_train, X_remaining, y_train, y_remaining = train_test_split(X, y, train_size=0.8, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_remaining, y_remaining, test_size=0.5, random_state=42)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test
    }
