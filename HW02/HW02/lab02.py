import math
from scipy import stats
from scipy.stats import norm, chi2
import tabulate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import bernoulli, binom
from IPython.display import HTML, display


def cdata01():
    #list of survey responses that are either happy, content, or sad.
    responses = ["happy", "happy", "content", "content", "content", "content", "happy", "content", "sad", "sad", "sad", "sad", "sad", "sad"]
    print(responses)
    survey_responses = pd.Categorical(responses, categories=["happy", "content", "sad"], ordered=True)
    print(survey_responses)
    print(type(survey_responses))

    #Create a pandas DataFrame
    df_survey_responses = pd.DataFrame({"response": survey_responses})
    print(df_survey_responses.head())

    #Descriptive Statistics
    print(df_survey_responses.describe())
    #sorting
    print(df_survey_responses.sort_values(by='response').head(10))
    print(df_survey_responses['response'].value_counts())
    print(df_survey_responses['response'].min())
    print(df_survey_responses['response'].max())

def cdata02(): #Bernoulli Random Variable
    p = 0.3
    X = bernoulli(p)
    print(np.round(X.pmf(1), 2))
    print(np.round(X.pmf(0), 2))
    X_samples = X.rvs(100000)
    sns.histplot(X_samples, stat="density", discrete=True, shrink=0.2);
    plt.show()
    print('Empirically calculated mean: {}'.format(X_samples.mean()))
    print('Theoretical mean: {}'.format(p))

    print('Empirically calculated standard deviation: {}'.format(X_samples.std()))
    print('Theoretical standard deviation: {}'.format((p * (1 - p)) ** (1 / 2)))

def cdata022():
    n = 6
    p = 0.3
    Y = bernoulli(p)
    Y_samples = [Y.rvs(1000000) for i in range(n)]
    Z_samples = sum(Y_samples)
    print('Empirically calculated expected value: {}'.format(Z_samples.mean()))
    print('Theoretical expected value: {}'.format(n * p))
    print('Empirically calculated variance: {}'.format(Z_samples.var()))
    print('Theoretical variance: {}'.format(n * p * (1 - p)))
    sns.histplot(Z_samples, stat="density", discrete=True, shrink=0.3);
    plt.show()

def cdata03(): # Goodness of Fit
    table = [["Day", 'M', 'T', 'W', 'T', 'F', 'S', 'S'],
             ["Expected (%)", 10, 10, 10, 20, 30, 15, 5],
             ["Observed", 30, 14, 34, 45, 57, 20, 10]]
    print(table)
    n = 7  # number of days in a week
    alpha = 0.05

    table = np.asarray(table)[1:, 1:]
    table = table.astype(np.float32)
    table[0] = table[0] / 100
    total_number_customers = np.sum(table[1])
    expected_num = table[0] * total_number_customers
    table = np.concatenate((table, expected_num.reshape(1, -1)))
    print(table)
    chi_sq_statistic = np.sum((table[2] - table[1]) ** 2 / table[2])
    print(chi_sq_statistic)
    if (1 - chi2.cdf(chi_sq_statistic, df=n - 1)) < alpha:
        print('Reject H_0')

def cdata04():  #Contingency Table Chi-square Test
    table = [['Sick', 15, 10, 30], ['Not sick', 100, 110, 90]]
    alpha = 0.05
    df = pd.DataFrame(table)
    df.columns = ['Effect', 'Pfizer', 'Janssen', 'Placebo']
    df = df.set_index('Effect')
    print(df)
    arr = df.to_numpy()
    arr = np.concatenate((arr, (arr.sum(axis=1)[0] / arr.sum() * arr.sum(axis=0)).reshape(1, -1)))
    arr = np.concatenate((arr, (arr.sum(axis=1)[1] / arr.sum() * arr.sum(axis=0)).reshape(1, -1)))
    print(arr)
    chi_sq_statistic = np.sum((arr[2] - arr[0]) ** 2 / arr[2]) + np.sum((arr[3] - arr[1]) ** 2 / arr[3])
    print(chi_sq_statistic)

    print('P-value = ' + str(np.round(1 - chi2.cdf(chi_sq_statistic, df=2 * 1), 4)))
    if 1 - chi2.cdf(chi_sq_statistic, df=2 * 1) < alpha:
        print('Reject H_0')