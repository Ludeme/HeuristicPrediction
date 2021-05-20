import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)


def outputRegressionPredictions(regressionPredictedValues, gameNamesTest, trainedModels, columnNames):
    resultsLocation = 'res/Output/PredictionResultsRegression.csv'

    # Write first row to results (learning model names)
    output = ""
    for model in trainedModels:
        for columnName in columnNames:
            output += type(model[0]).__name__ + "-" + columnName + ","
    f = open(resultsLocation, "w")
    f.write("GameName," + output[0:-1] + ",Average\n")

    # Test data predictions
    outputString = ""
    print("\nPredicting test games.")
    for rowIndex in range(len(regressionPredictedValues)):

        # Game name
        outputRow = gameNamesTest[rowIndex] + ","

        # Regression predictions
        for modelIndex in range(len(regressionPredictedValues[rowIndex])):
            for columnIndex in range(len(regressionPredictedValues[rowIndex][modelIndex])):
                outputRow += str(regressionPredictedValues[rowIndex][modelIndex][columnIndex]) + ","

        # Average predicted value
        predictedValuesList = outputRow.split(",")[1:-1]
        totalValue = 0
        for i in predictedValuesList:
            totalValue += int(float(i))
        averageValue = totalValue / len(predictedValuesList)
        outputRow += str(averageValue) + ","

        outputString += outputRow + "\n"

    f = open(resultsLocation, "a")
    f.write(outputString)
    f.close()