# possible algorithms
from sklearn.decomposition import PCA, SparsePCA, TruncatedSVD, FactorAnalysis, FastICA, NMF

from low_signal_autoencoder import LowSignalAutoencoder

pca = PCA(n_components=1)
spca = SparsePCA(n_components=1, alpha=1, ridge_alpha=0.01,
                 max_iter=1000, tol=0.001, method='lars')
tsvd = TruncatedSVD(n_components=1)
fa = FactorAnalysis(n_components=1, tol=0.001)
ica = FastICA(n_components=1, random_state=0)
nmf = NMF(n_components=1)
autoencoder = LowSignalAutoencoder(image_size=56, num_epochs=200, dense_layer_neurons=2, show_loss_plot=False)


def get_algorithms():
    """
    :return: List of all algorithms
    """
    return [
        ('Principal Component Analysis', 'pca', pca),
        ('Sparse Principal Component Analysis', 'spca', spca),
        ('Truncated Singular Value Decomposition', 'tsvd', tsvd),
        ('Factor Analysis', 'fa', fa),
        ('Independent Component Analysis', 'ica', ica),
        ('Non-negative Matrix Factorization', 'nmf', nmf),
        ('Autoencoder', 'autoencoder', autoencoder)
    ]


def get_algorithm(algorithm: str):
    """
    Returns a scikit learn algorithm based on name or abbreviation.

    :param algorithm: The name or abbreviation of the algorithm to use.
    :return: Instance of scikit learn algorithm with preset hyperparameters.
    """
    for name, abbr, alg in get_algorithms():
        if algorithm == name or algorithm == abbr:
            return alg


def get_algorithm_name(algorithm):
    """
    Returns the name of a scikit learn algorithm.

    :param algorithm: Instance of scikit-learn algorithm
    :return: Full name of algorithm
    """
    for name, abbr, alg in get_algorithms():
        if algorithm == abbr or algorithm == alg:
            return name


def is_autoencoder(algorithm):
    """
    Returns true if the algorithm passed is an autoencoder.

    :param algorithm: The algorithm to check

    :return: True if the algorithm is an autoencoder, false otherwise
    """
    return type(algorithm) == LowSignalAutoencoder
