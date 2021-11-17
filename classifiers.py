from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import DotProduct, RBF
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from KerasClassifier import KerasMLP
import numpy as np


def get_classifier(name="SVM"):
    """Prepare a Classifier that corresponds to the given name.
    Inner CV and Hyperparameter Optimization has to be defined here."""
    inner_cv = StratifiedKFold(n_splits=5)
    resampler = SMOTE()
    pipeline = Pipeline(steps=[
        ("scaler", MinMaxScaler()),
        ("pca", PCA(n_components="mle")),
        ("resampler", resampler)
    ])
    if name == "SVM":

        pipeline.steps.append(("svm", SVC(probability=True, kernel='rbf')))
        hyperparameters = {
            "svm__C"    : np.logspace(-5, 5, num=5),
            "svm__gamma": np.logspace(-5, 5, num=5)
        }
        return GridSearchCVWrapper(
            pipeline,
            hyperparameters,
            cv=inner_cv,
            verbose=1,
            refit=True
        )
    elif name == "Ada":
        pipeline.steps.append(("ada", AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, criterion="entropy", max_features=None, splitter="best"), algorithm="SAMME", n_estimators=50)))
        hyperparameters = {
            "ada__n_estimators": [5, 10, 25, 50, 100, 200]
        }
        return GridSearchCVWrapper(
            pipeline,
            hyperparameters,
            cv=inner_cv,
            verbose=1,
            refit=True
        )
    elif name == "CatBoost":
        pipeline.steps.append(("catboost", CatBoostClassifier()))
        return pipeline
    elif name == "MLP - SKLearn":
        pipeline.steps.append(("mlp", MLPClassifier()))
        hyperparameters = {
            "mlp__hidden_layer_sizes": [(50), (30), (20, 10)]
        }
        return GridSearchCVWrapper(
            pipeline,
            hyperparameters,
            cv=inner_cv,
            verbose=1,
            refit=True
        )
    elif name == "MLP - Tensorflow":
        pipeline.steps.append(("mlp", KerasMLP()))
        hyperparameters = {}
        return GridSearchCVWrapper(
            pipeline,
            hyperparameters,
            cv=inner_cv,
            verbose=1,
            refit=True
        )
    elif name == "KNN":
        pipeline.steps.append(("knn", KNeighborsClassifier(metric="euclidean")))
        hyperparameters = {
            "knn__n_neighbors": [3, 5, 9, 13, 19]
        }
        return GridSearchCVWrapper(
            pipeline,
            hyperparameters,
            cv=inner_cv,
            verbose=1,
            refit=True
        )
    elif name == "Gaussian Process":
        pipeline.steps.append(("gp", GaussianProcessClassifier()))
        hyperparameters = {
            "kernel": [DotProduct(), RBF()]
        }
        return GridSearchCVWrapper(
            pipeline,
            hyperparameters,
            cv=inner_cv,
            verbose=1,
            refit=True
        )

class GridSearchCVWrapper(GridSearchCV):
    def fit(self, X, y=None, groups=None, **fit_params):
        super().fit(X, y=y, groups=groups, **fit_params)
        print(f"Best Parameters: {self.best_params_}")
