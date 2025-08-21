import pandas as pd
import matplotlib.pyplot as plt
import numpy


def plot_scatter_line(df, a, b, x="x", y="y"):
    # Scatterplot;
    plt.scatter(df[x], df[y])

    # Set data for line;
    x_vals = numpy.linspace(df[x].min(), df[x].max(), 100)
    y_vals = a * x_vals + b

    plt.plot(x_vals, y_vals, color="red", label=f"Line: y = {a}x + {b}")

    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"{x}-{y} data")
    plt.grid()

    plt.show()


def costFunc(data_x: numpy.array, data_y: numpy.array, theta: numpy.array) -> float:
    tmp = theta[0] + theta[1] * data_x - data_y

    return numpy.inner(tmp, tmp) / (2 * data_x.shape[0])
