import numpy
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from tools.plot import plot_scatter_line, costFunc

df = pd.read_csv("./dataset/Boston.csv")
dataset = df[["medv", "rm"]].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(dataset[:, 0], dataset[:, 1])

def update_params(data_x: numpy.array, data_y: numpy.array, theta: numpy.array, alpha):
    tmp_vec = theta[0] + theta[1] * data_x - data_y
    return theta

update_params(x_train, y_train, numpy.array([0, 0]), 0.1)