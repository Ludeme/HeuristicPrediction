import os
import pandas as pd


# Record the input data for without labels
def getUnlabelledData(inputDataLocation):
    csvInputData = pd.read_csv(inputDataLocation, sep=',', header=0)
    inputFeatures = csvInputData.iloc[:, 1:]
    firstColumn = csvInputData.iloc[:, 0]
    print("\nInput CSV:\n", csvInputData, "\n")
    return inputFeatures, firstColumn


# Record the input data including labels
# Merges the two csvs to ensure consistency
def GetLabelledData(trainingFeaturesLocation, trainingLabelsLocation):
    inputFeatures = pd.read_csv(trainingFeaturesLocation, sep=',', header=0).iloc[:, 1:]
    inputLabels = pd.read_csv(trainingLabelsLocation, sep=',', header=0).iloc[:, 1:]
    firstColumn = pd.read_csv(trainingFeaturesLocation, sep=',', header=0).iloc[:, 0]
    print("\nInput Features:\n", inputFeatures, "\n")
    print("\nInput Labels:\n", inputLabels, "\n")
    return inputFeatures, inputLabels, firstColumn
