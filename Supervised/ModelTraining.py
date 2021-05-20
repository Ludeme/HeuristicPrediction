import warnings
from copy import deepcopy
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)


def getModelPredictions(models, inputFeaturesTest):
    predictionResults = []
    for rowIndex in range(inputFeaturesTest.shape[0]):
        predictionResults.append([])
        inputFeaturesTestNumpy = inputFeaturesTest.to_numpy()
        for modelIndex in range(len(models)):
            predictionResults[-1].append([])
            for columnIndex in range(len(models[modelIndex])):
                predictionResults[-1][-1].append([])
                model = models[modelIndex][columnIndex]
                inputFeatures = inputFeaturesTestNumpy[rowIndex].reshape(1, -1)
                predictedValue = model.predict(inputFeatures)
                predictionResults[rowIndex][modelIndex][columnIndex] = predictedValue[0]
    return predictionResults


# Return a nested array of trained models [algorithmIndex][heuristicIndex]
def trainModels(inputFeatures, allLabels, models):

    for modelIndex in range(len(models)):
        modelsAcrossAllLabels = []
        model = models[modelIndex]
        modelMAE = []
        print("\n", type(model).__name__)

        for i in range(allLabels.shape[1]):
            labels = allLabels.iloc[:, i]

            # Cross Validation scores (LOOCV)
            cv = KFold(n_splits=len(labels))
            scores = cross_val_score(model, inputFeatures, labels, scoring="neg_mean_absolute_error", cv=cv)
            modelMAE.append(scores.mean())

            # Train model
            trainedModel = deepcopy(model)
            trainedModel.fit(inputFeatures, labels)
            modelsAcrossAllLabels.append(deepcopy(trainedModel))

        models[modelIndex] = modelsAcrossAllLabels
        print("MAE ", np.average(modelMAE), " (", np.std(modelMAE), ")", sep="")

    return models
