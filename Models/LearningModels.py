from sklearn import linear_model, svm, tree, neighbors, ensemble, neural_network
from sklearn.dummy import DummyRegressor

# Learning models (Regression)
regressionModels = [
    DummyRegressor(strategy="mean"),
    linear_model.Ridge(),
    linear_model.Lasso(),
    linear_model.ElasticNet(),
    linear_model.BayesianRidge(),
    ensemble.RandomForestRegressor(),
    ensemble.GradientBoostingRegressor(),
    tree.DecisionTreeRegressor(),
    neighbors.KNeighborsRegressor(),
    svm.LinearSVR(random_state=0),
    neural_network.MLPRegressor(),
]
