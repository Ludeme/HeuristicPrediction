from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Scale the values for each feature to between 0 and 1
def scaleInputFeatures(inputData):
    min_max_scaler = MinMaxScaler()
    inputDataOriginal = deepcopy(inputData)
    inputData = min_max_scaler.fit_transform(inputData)
    inputData = pd.DataFrame(inputData, index=inputDataOriginal.index, columns=inputDataOriginal.columns)
    return inputData


# Shuffle input features and labels
def shuffleInput(inputFeatures, labels):
    np.random.seed(0)
    inputFeaturesNumpy = inputFeatures.to_numpy()
    labelsNumpy = labels.to_numpy()
    assert len(inputFeaturesNumpy) == len(labelsNumpy)
    p = np.random.permutation(len(inputFeaturesNumpy))
    inputFeaturesShuffled = inputFeaturesNumpy[p]
    labelsShuffled = labelsNumpy[p]
    return inputFeaturesShuffled, labelsShuffled


def getPreprocessedInputData(inputData, featureSelector):
    inputData.fillna(0, inplace=True)                                               # replaces all NaN values with a default
    inputData = pd.get_dummies(inputData)                                           # Convert categorical data to boolean data (one-hot encoding)
    inputData = scaleInputFeatures(inputData)                                       # Scale the values for each feature to between 0 and 1
    if featureSelector is not None:
        inputData = featureSelector.fit_transform(inputData)                            # Transform the input data with the specified feature selector
    return inputData

