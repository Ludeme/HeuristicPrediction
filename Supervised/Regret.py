from copy import deepcopy
import numpy as np
import pandas as pd
import Utils.Preprocessing as Preprocessing


# Calculates the regret for the models.
def calculateCrossValidationRegretRegression(inputFeatures, labels, regressionModels):

    # Shuffle inputFeatures
    inputFeaturesShuffled, labelsShuffled = Preprocessing.shuffleInput(inputFeatures, labels)

    for model in regressionModels:
        totalRegret = 0

        for leaveOneOutIndex in range(len(inputFeaturesShuffled)):
            inputFeaturesTrain = np.delete(deepcopy(inputFeaturesShuffled), leaveOneOutIndex, axis=0)
            labelsTrain = np.delete(deepcopy(labelsShuffled), leaveOneOutIndex, axis=0)
            inputFeaturesTest = deepcopy(inputFeaturesShuffled)[leaveOneOutIndex]
            labelsTest = deepcopy(labelsShuffled)[leaveOneOutIndex]

            # Determine the best predicted heuristics for the game in the test set
            bestHeuristicIndex = -1
            bestValue = -999999
            for i in range(labelsTrain.shape[1]):
                labelsTrainColumn = labelsTrain[:,i]
                model.fit(inputFeaturesTrain, labelsTrainColumn)
                train_pred = model.predict([inputFeaturesTest])[0]
                if train_pred > bestValue:
                    bestValue = train_pred
                    bestHeuristicIndex = i

            # Calculate the best possible heuristics for the game in the test set
            predictedHeuristicValue = labelsShuffled[leaveOneOutIndex][bestHeuristicIndex]
            actualBestHeuristicValue = max(labelsTest)
            totalRegret += actualBestHeuristicValue - predictedHeuristicValue

        totalRegret = totalRegret / len(inputFeaturesShuffled)

        print("\n", type(model).__name__)
        print("Regret:", totalRegret)
