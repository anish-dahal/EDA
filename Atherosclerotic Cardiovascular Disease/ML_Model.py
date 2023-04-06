# Import libraries
import pandas as pd
import numpy as np

import pickle
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    MinMaxScaler,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

# Class data_processing
class DataProcessing:
    """
    This class take dataframe and do dataprocessing
    like remove null value, outliers and encoding process.
    """

    def __init__(self, df):
        self.__df = df

    def data_type(self):
        return self.__df.dtypes

    def shape(self):
        return self.__df.shape

    def show(self):
        return self.__df.head()

    def preprocessing(self):
        # remove duplicates rows
        self.__df.drop_duplicates(inplace=True)

        # Remove null value
        self.__df.dropna(inplace=True)

        # Remove outliers on the 'Age' column
        self.__df = self.__df[(self.__df["Age"] >= 0) & (self.__df["Age"] < 100)]

        # make categorical data into category datatype
        for column in self.__df.columns:
            if self.__df[column].nunique() < 5:
                self.__df[column] = self.__df[column].astype("category")

        # label encoding all category data except the 'Sex' and 'Risk' column
        label_encoding_column = (
            (self.__df.drop(columns=["Sex", "Risk"])).select_dtypes("category").columns
        )
        self.__df[label_encoding_column] = self.__df[label_encoding_column].apply(
            LabelEncoder().fit_transform
        )

    def clean_dataframe(self):
        return self.__df


class SklearnModel:
    """
    used to select 3 different machine learning algorithm from sklearn
    """

    def __init__(self):
        pass

    def logistic_regression(
        self,
        penalty="l2",
        dual=False,
        tol=0.0001,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=99,
        solver="lbfgs",
        max_iter=1000,
        multi_class="auto",
        verbose=0,
        warm_start=False,
        n_jobs=-1,
        l1_ratio=None,
    ):
        self.Classifier = LogisticRegression(
            penalty=penalty,
            dual=dual,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs,
            l1_ratio=l1_ratio,
        )

    def svm(self, **parameters):
        kernel = parameters.get("kernel", "rbf")
        probability = parameters.get("probability", True)
        tol = parameters.get("tol", 0.001)
        class_weight = parameters.get("class_weight", None)
        verbose = parameters.get("verbose", False)
        max_iter = parameters.get("max_iter", -1)
        decision_function_shape = parameters.get("decision_function_shape", "ovr")
        random_state = parameters.get("random_state", 99)
        self.Classifier = SVC(
            kernel=kernel,
            probability=probability,
            tol=tol,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            random_state=random_state,
        )

    def decision_tree(self, **parameters):
        criterion = parameters.get("criterion", "gini")
        max_depth = parameters.get("max_depth", None)
        min_samples_split = parameters.get("min_samples_split", 2)
        min_samples_leaf = parameters.get("min_samples_leaf", 1)
        random_state = parameters.get("random_state", 99)
        self.Classifier = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

    def pipeline(self, numeric_column, category_column):
        """
        make pipeline of onehotencoder and MinMaxScalar
        with one of the selected machine learning algorithm

        Parameters
        ----------
        numeric_column : list
            dataset column name which data type is int or float
            list of string
        category_column : list
            dataset column name which data type is object
            list of string
        """
        column_trans = ColumnTransformer(
            transformers=[
                ("ohe", OneHotEncoder(), category_column),
                ("MinMaxScaler", MinMaxScaler(), numeric_column),
            ]
        )

        self.model = Pipeline(
            [("transformer", column_trans), ("classifier", self.Classifier)]
        )

    def fit(self, X, y):
        (self.model).fit(X, y)

    def predict(self, X):
        return (self.model).predict(X)

    def predict_prob(self, X):
        return (self.model).predict_proba(X)


class Metrics:
    """
    used to find different metrics from sklearn
    method will take target true value and
    target predicted value or target predicted value probability
    """

    def __init__(self):
        pass

    def confusion_matrix(self, y, y_pred):
        return confusion_matrix(y, y_pred)

    def accuracy(self, y, y_pred):
        return accuracy_score(y, y_pred)

    def precision_recall_fscore(self, y, y_pred):
        (
            model_precision,
            model_recall,
            model_f1_score,
            _,
        ) = precision_recall_fscore_support(y, y_pred, average="macro")
        return (model_precision, model_recall, model_f1_score)

    def roc_auc_value(self, y, y_prob, classes=None):
        return roc_auc_score(
            y, y_prob, labels=classes, multi_class="ovr", average="macro"
        )

    def roc_plot(self, y, y_prob, classes=None):
        """
        plot roc curve for multiclass target using one vs rest

        Parameters
        ----------
        y : array (n,)
            array of target class
        y_prob : array (n, no_of_class)
            probability of target classes
        classes : lisk, optional
            list of string
            used to pass target class name to identify labels in the plot, by default None
        """
        fpr = {}
        tpr = {}
        if not classes:
            classes = range(y_prob.shape[1])
        n_class = len(classes)

        for i in range(n_class):
            fpr[i], tpr[i], _ = roc_curve(y, y_prob[:, i], pos_label=i)

        # plotting
        colour = [
            "red",
            "blue",
            "green",
            "yellow",
            "purple",
            "orange",
            "white",
            "black",
        ]
        for i in range(n_class):
            plt.plot(
                fpr[i],
                tpr[i],
                linestyle="--",
                color=colour[i],
                label=f"{classes[i]} vs Rest",
            )
        plt.title("Multiclass ROC curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive rate")
        plt.legend(loc="best")
        plt.show()


# LogisticRegression from scratch
class Logistic_Regression:
    def __init__(self, max_iter=200, lr=0.0001):
        """
        used to pass value for logistic regressing while making object

        Parameters
        ----------
        max_iter : int, optional
            number of iteration to optimize parameter, by default 200
        lr : float, optional
            learning rate, by default 0.0001
        """
        self.max_iter = max_iter
        self.lr = lr

    @staticmethod
    def _softmax(Z):
        return np.array([np.exp(z) / np.sum(np.exp(z)) for z in Z])

    def loss_function(self, y_true, y_pred):
        return np.sum(np.square(y_true - y_pred)) / (2 * y_true.size)

    @staticmethod
    def _add_intercept(X):
        """
        add intercept to make linear regressing equation
        y = W.X + b where W and b is parameter

        Parameters
        ----------
        X : array

        Returns
        -------
        array
        """
        intercept = np.ones(shape=(X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def fit(self, X, y):
        """
        used to fit logistic regression model

        Parameters
        ----------
        X : array, DataFrame, or series
            value is numeric
        y : array, series
            contain label encoded value
        """
        y = y.to_numpy(dtype=int)
        # use one hot encoder if label encoding is only done.
        if y.shape[0] == y.size:
            y_encoded = np.zeros((y.size, y.max() + 1), dtype=int)
            # replacing 0 with a 1 at the index of the original array
            y_encoded[np.arange(y.size), y] = 1
            y = y_encoded

        # add intercept for input data
        X = self._add_intercept(X)

        # initalize theta with random value
        np.random.seed(55)
        self.theta = np.random.rand(X.shape[1], y.shape[1])

        for _ in range(self.max_iter):
            z = np.dot(X, self.theta)
            h = self._softmax(z)
            self.theta -= self.lr * np.dot(X.T, (h - y)) / y.size
            self.loss = self.loss_function(y, h)

    def predict_prob(self, X):
        X = self._add_intercept(X)
        return self._softmax(np.dot(X, self.theta))

    def predict(self, X):
        pred_prob = self.predict_prob(X)
        pred_argmax = np.argmax(pred_prob, axis=1)
        return pred_argmax


def save_model(model, path):
    """
    save the model for future use file in .pkl

    Parameters
    ----------
    model : object
        the model which is fit with training data
    path : str
        path to save the model
    """
    pickle.dump(model, open(path, "wb"))


def load_model(path):
    """
    load model from the provided location

    Parameters
    ----------
    path : str

    Returns
    -------
    object
        the model that was saved in .pkl format
    """
    return pickle.load(open(path, "rb"))
