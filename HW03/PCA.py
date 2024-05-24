import numpy as np
import pandas as pd


def PCA(X, num_components):
    # Step-1 Subtract the mean of each variable
    X_meaned = X - np.mean(X, axis=0)
    print('mean subtracted data')
    print(X_meaned)


    # Step-2  Calculate the Covariance Matrix
    cov_mat = np.cov(X_meaned, rowvar=False)
    print('Covariance Matrix')
    print(cov_mat)

    # Step-3 Compute the Eigenvalues and Eigenvectors
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

    print('eigen_values')
    print(eigen_values)

    print('eigen_vectors')
    print(eigen_vectors)


    # Step-4 Sort Eigenvalues in descending order
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]
    print('sorted_eigenvalue')
    print(sorted_eigenvalue)

    print('sorted_eigenvectors')
    print(sorted_eigenvectors)

    # Step-5 Select a subset from the rearranged Eigenvalue matrix
    eigenvector_subset = sorted_eigenvectors[:, 0:num_components]

    # Step-6  Transform the data
    X_reduced = np.dot(eigenvector_subset.transpose(), X_meaned.transpose()).transpose()
    print('Transformed data')
    print(X_reduced)
    return X_reduced

def testPCA():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    data = pd.read_csv(url, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'target'])

    print(data)

    # prepare the data
    x = data.iloc[:, 0:4]
    print(x)
    # prepare the target
    target = data.iloc[:, 4]

    # Applying it to PCA function
    mat_reduced = PCA(x, 2)
    print(mat_reduced)
    # Creating a Pandas DataFrame of reduced Dataset
    principal_df = pd.DataFrame(mat_reduced, columns=['PC1', 'PC2'])

    # Concat it with target variable to create a complete Dataset
    principal_df = pd.concat([principal_df, pd.DataFrame(target)], axis=1)
    print(principal_df)

if __name__ == '__main__':
    testPCA()