import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Utils.Preprocessing as Preprocessing


# Plots the inputData after feature selection, using the provided set of labels
def showPlot(inputData, colourLabels):

    # Set the plot style
    plt.style.use('bmh') # seaborn classic bmh
    colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Plot labels (2D)
    if inputData.shape[1] == 2:
        for i in np.unique(colourLabels):
            condition = colourLabels == i
            plt.scatter(inputData[condition, 0], inputData[condition, 1], label=i, color=colours[i])
        plt.legend()
        plt.show()
    else:
        print("Error! The inputData must be reduced to 2 dimensions.")


# Records the labels for a set of game names, based on the csv stored at labelLocation
def recordLabels(labelLocation, gameNames):
    labels = []
    labelNames = ["None"]

    # Read in data from external csv
    csvInputDataLabels = pd.read_csv(labelLocation, sep=',', header=0)
    gameNamesLabels = csvInputDataLabels.iloc[:, 0].tolist()
    inputDataLabels = csvInputDataLabels.iloc[:, 1:]
    for i in range(len(gameNames)):
        if gameNames[i] in gameNamesLabels:
            rowIndex = gameNamesLabels.index(gameNames[i])
            categoryName = inputDataLabels.iloc[rowIndex, 0]
            if categoryName not in labelNames:
                labelNames.append(categoryName)
            labels.append(labelNames.index(categoryName))
        else:
            labels.append(0)
    return labels


def plotDataCategories(inputData, gameNames, labelColoursLocation):
    colourLabels = recordLabels(labelColoursLocation, gameNames)
    showPlot(inputData, colourLabels)
