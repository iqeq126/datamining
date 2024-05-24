import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from statsmodels import robust
import scipy.stats as stats
import random





def prob01():
    num_throws = 10000
    outcomes = np.zeros(num_throws)
    for i in range(num_throws):
        # let's roll the die
        outcome = np.random.choice(['1', '2', '3', '4', '5', '6'])
        outcomes[i] = outcome

    val, cnt = np.unique(outcomes, return_counts=True)
    prop = cnt / len(outcomes)
    print(prop)

    # Now that we have rolled our die 10000 times, let's plot the results
    plt.bar(val, prop)
    plt.ylabel("Probability")
    plt.xlabel("Outcome")
    plt.show()

def prob02():
    rolls = np.random.randint(1, 7, 20)
    val, counts = np.unique(rolls, return_counts=True)
    plt.stem(val, counts / len(rolls), basefmt="C2-", use_line_collection=True)
    plt.show()

    throws = np.random.randint(1, 7, 100000)
    val, counts = np.unique(throws, return_counts=True)
    plt.stem(val, counts / len(throws), basefmt="C2-", use_line_collection=True)
    plt.show()

    np.random.seed(123)
    data = np.random.normal(0.3, 0.1, 1000) #a normal distribution with a mean of 0.3 and a standard deviation of 0.1
    hist = plt.hist(data, bins=13, range=(-0.3, 1))
    plt.show()

    hist = plt.hist(data, bins=24, range=(-0.2, 1), density=True)
    plt.show()


def prob03():
    # Normal(Gaussian)   Distribution
    n = np.arange(-50, 50)
    mean = 0
    normal = stats.norm.pdf(n, mean, 10)
    plt.plot(n, normal)
    plt.xlabel('Distribution', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title("Normal Distribution")
    plt.show()

    # Binomial Distribution
    for prob in range(3, 10, 3):
        x = np.arange(0, 25)
        binom = stats.binom.pmf(x, 20, 0.1 * prob)
        plt.plot(x, binom, '-o', label="p = {:f}".format(0.1 * prob))
        plt.xlabel('Random Variable', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        plt.title("Binomial Distribution varying p")
        plt.legend()
    plt.show()

    #Poisson Distribution
    # n = number of events, lambd = expected number of events
    # which can take place in a period
    for lambd in range(2, 8, 2):
        n = np.arange(0, 10)
        poisson = stats.poisson.pmf(n, lambd)
        plt.plot(n, poisson, '-o', label="λ = {:f}".format(lambd))
        plt.xlabel('Number of Events', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        plt.title("Poisson Distribution varying λ")
        plt.legend()
    plt.show()

    #Exponential Distribution
    for lambd in range(1, 10, 3):
        x = np.arange(0, 15, 0.1)
        y = 0.1 * lambd * np.exp(-0.1 * lambd * x)
        plt.plot(x, y, label="λ = {:f}".format(0.1 * lambd))
        plt.xlabel('Random Variable', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        plt.title("Exponential Distribution varying λ")
        plt.legend()
    plt.show()





def dataA01():  #basic info about the data
    # Load Iris.csv into a pandas dataFrame.
    iris = pd.read_csv("iris.csv")
    # how many data-points and features?
    print(iris.shape)
    # What are the column names in our dataset?
    print(iris.columns)

    print(iris.describe(include='all'))

    # (Q) How many data points for each class are present?
    # (or) How many flowers for each species are present?

    print(iris["species"].value_counts())
    # balanced-dataset vs imbalanced datasets
    # Iris is a balanced dataset as the number of data points for every class is 50.

    # 2-D scatter plot:
    # ALWAYS understand the axis: labels and scale.

    iris.plot(kind='scatter', x='sepal_length', y='sepal_width');
    plt.show()

    # cannot make much sense out it.
    # What if we color the points by thier class-label/flower-type.
    # 2-D Scatter plot with color-coding for each flower type/class.
    # Here 'sns' corresponds to seaborn.
    sns.set_style("whitegrid");
    sns.FacetGrid(iris, hue="species", height=4) \
        .map(plt.scatter, "sepal_length", "sepal_width") \
        .add_legend();
    plt.show();

    # Notice that the blue points can be easily seperated
    # from red and green by drawing a line.
    # But red and green data points cannot be easily seperated.
    # Can we draw multiple 2-D scatter plots for each combination of features?

def dataA02():
    """ Histogram, PDF, CDF"""
    # Load Iris.csv into a pandas dataFrame.
    iris = pd.read_csv("iris.csv")
    # What about 1-D scatter plot using just one feature?
    # 1-D scatter plot of petal-length

    iris_setosa = iris.loc[iris["species"] == "setosa"];
    iris_virginica = iris.loc[iris["species"] == "virginica"];
    iris_versicolor = iris.loc[iris["species"] == "versicolor"];
    # print(iris_setosa["petal_length"])
    plt.plot(iris_setosa["petal_length"], np.zeros_like(iris_setosa['petal_length']), 'o')
    plt.plot(iris_versicolor["petal_length"], np.zeros_like(iris_versicolor['petal_length']), 'o')
    plt.plot(iris_virginica["petal_length"], np.zeros_like(iris_virginica['petal_length']), 'o')

    plt.show()
    # Disadvantages of 1-D scatter plot: Very hard to make sense as points
    # are overlapping a lot.
    # Are there better ways of visualizing 1-D scatter plots?

    # Need for Cumulative Distribution Function (CDF)
    # We can visually see what percentage of versicolor flowers have a
    # petal_length of less than 5?
    # How to construct a CDF?
    # How to read a CDF?

    # Plot CDF of petal_length

    counts, bin_edges = np.histogram(iris_setosa['petal_length'], bins=10,
                                     density=True)
    pdf = counts / (sum(counts))
    print(pdf);
    print(bin_edges);
    cdf = np.cumsum(pdf)  # cumulative sum of PDF
    plt.plot(bin_edges[1:], pdf);
    plt.plot(bin_edges[1:], cdf)

    counts, bin_edges = np.histogram(iris_setosa['petal_length'], bins=20,
                                     density=True)
    pdf = counts / (sum(counts))
    plt.plot(bin_edges[1:], pdf);

    plt.show();

    # Need for Cumulative Distribution Function (CDF)
    # We can visually see what percentage of versicolor flowers have a
    # petal_length of less than 1.6?
    # How to construct a CDF?
    # How to read a CDF?

    # Plot CDF of petal_length

    counts, bin_edges = np.histogram(iris_setosa['petal_length'], bins=10,
                                     density=True)
    pdf = counts / (sum(counts))
    print(pdf);
    print(bin_edges)

    # compute CDF
    cdf = np.cumsum(pdf)
    plt.plot(bin_edges[1:], pdf)
    plt.plot(bin_edges[1:], cdf)

    plt.show();

    # Plots of CDF of petal_length for various types of flowers.

    # Misclassification error if you use petal_length only.

    counts, bin_edges = np.histogram(iris_setosa['petal_length'], bins=10,
                                     density=True)
    pdf = counts / (sum(counts))
    print(pdf);
    print(bin_edges)
    cdf = np.cumsum(pdf)
    plt.plot(bin_edges[1:], pdf)
    plt.plot(bin_edges[1:], cdf)

    # virginica
    counts, bin_edges = np.histogram(iris_virginica['petal_length'], bins=10,
                                     density=True)
    pdf = counts / (sum(counts))
    print(pdf);
    print(bin_edges)
    cdf = np.cumsum(pdf)
    plt.plot(bin_edges[1:], pdf)
    plt.plot(bin_edges[1:], cdf)

    # versicolor
    counts, bin_edges = np.histogram(iris_versicolor['petal_length'], bins=10,
                                     density=True)
    pdf = counts / (sum(counts))
    print(pdf);
    print(bin_edges)
    cdf = np.cumsum(pdf)
    plt.plot(bin_edges[1:], pdf)
    plt.plot(bin_edges[1:], cdf)

    plt.show();

def dataA03(): #Mean, Variance and Std-dev, (Median, Percentile, Quantile, IQR, MAD)
    # Mean, Variance, Std-deviation,

    # Load Iris.csv into a pandas dataFrame.
    iris = pd.read_csv("iris.csv")
    # What about 1-D scatter plot using just one feature?
    # 1-D scatter plot of petal-length

    iris_setosa = iris.loc[iris["species"] == "setosa"];
    iris_virginica = iris.loc[iris["species"] == "virginica"];
    iris_versicolor = iris.loc[iris["species"] == "versicolor"];

    print("Means:")
    print(np.mean(iris_setosa["petal_length"]))
    # Mean with an outlier.
    print(np.mean(np.append(iris_setosa["petal_length"], 50)));
    print(np.mean(iris_virginica["petal_length"]))
    print(np.mean(iris_versicolor["petal_length"]))

    print("\nStd-dev:");
    print(np.std(iris_setosa["petal_length"]))
    print(np.std(iris_virginica["petal_length"]))
    print(np.std(iris_versicolor["petal_length"]))
    # Median, Quantiles, Percentiles, IQR.
    print("\nMedians:")
    print(np.median(iris_setosa["petal_length"]))
    # Median with an outlier
    print(np.median(np.append(iris_setosa["petal_length"], 50)));
    print(np.median(iris_virginica["petal_length"]))
    print(np.median(iris_versicolor["petal_length"]))

    print("\nQuantiles:")
    print(np.percentile(iris_setosa["petal_length"], np.arange(0, 100, 25)))
    print(np.percentile(iris_virginica["petal_length"], np.arange(0, 100, 25)))
    print(np.percentile(iris_versicolor["petal_length"], np.arange(0, 100, 25)))

    print("\n90th Percentiles:")
    print(np.percentile(iris_setosa["petal_length"], 90))
    print(np.percentile(iris_virginica["petal_length"], 90))
    print(np.percentile(iris_versicolor["petal_length"], 90))


    print("\nMedian Absolute Deviation")
    print(robust.mad(iris_setosa["petal_length"]))
    print(robust.mad(iris_virginica["petal_length"]))
    print(robust.mad(iris_versicolor["petal_length"]))