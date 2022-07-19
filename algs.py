# possible algorithms
from sklearn.decomposition import PCA, SparsePCA, TruncatedSVD, FactorAnalysis, FastICA, NMF

pca = PCA(n_components=1)
spca = SparsePCA(n_components=1, alpha=1, ridge_alpha=0.01,
                 max_iter=1000, tol=0.001, method='lars')
tsvd = TruncatedSVD(n_components=1)
fa = FactorAnalysis(n_components=1, tol=0.001)
ica = FastICA(n_components=1, random_state=0)
nmf = NMF(n_components=1)


def get_algorithms():
    """Returns a list of all algorithms used"""
    return [
        ('Principal Component Analysis', 'pca', pca),
        ('Sparse Principal Component Analysis', 'spca', spca),
        ('Truncated Singular Value Decomposition', 'tsvd', tsvd),
        ('Factor Analysis', 'fa', fa),
        ('Independent Component Analysis', 'ica', ica),
        ('Non-negative Matrix Factorization', 'nmf', nmf)
    ]


def get_algorithm(algorithm):
    """
    Summary:
        Uses algs.py to return a scikit learn algorithm based on name or abbreviation

    Args:
        algorithm (str): The name or abbreviation of the algorithm to use

    Returns:
        algorithm: Instance of scikit learn algorithm with set hyperparameters
    """
    for name, abbr, alg in get_algorithms():
        if algorithm == name or algorithm == abbr:
            return alg


def get_algorithm_name(algorithm):
    """
    Summary:
        Uses algs.py to return a scikit learn algorithm based on name or abbreviation

    Args:
        algorithm (str): The abbreviation or instance of the algorithm to use

    Returns:
        algorithm: Unabbreviated name of algorithm
    """
    for name, abbr, alg in get_algorithms():
        if algorithm == abbr or algorithm == alg:
            return name
