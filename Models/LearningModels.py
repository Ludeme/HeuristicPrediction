from sklearn import linear_model, svm, tree, neighbors, ensemble, neural_network
from sklearn.dummy import DummyRegressor

# Learning models (Regression)
regressionModels = [
    DummyRegressor(strategy="mean"),
    linear_model.Ridge(random_state=0),
    linear_model.Lasso(random_state=0),
    linear_model.ElasticNet(random_state=0),
    linear_model.BayesianRidge(),
    ensemble.RandomForestRegressor(random_state=0),
    ensemble.GradientBoostingRegressor(random_state=0),
    tree.DecisionTreeRegressor(random_state=0),
    neighbors.KNeighborsRegressor(),
    svm.LinearSVR(random_state=0),
    neural_network.MLPRegressor(random_state=0),
]
