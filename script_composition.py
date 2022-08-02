import functools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Callable

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ComposableFunction = Callable[[np.ndarray], np.ndarray]

def compose(*functions: ComposableFunction) -> ComposableFunction:
    """
    Compose a list of functions into a single function.
    """
    return functools.reduce(lambda f, g: lambda x: g(f(x)), functions)


def scale(X: np.ndarray) -> np.ndarray:
    """
    Scale the data to have mean 0 and variance 1.
    """
    return 