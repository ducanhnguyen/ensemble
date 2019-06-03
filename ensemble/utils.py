from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def load_data():
    '''
    Classes 10
    Samples per class ~180
    Samples total 1797
    Dimensionality 64
    Features integers 0-16
    :return:
    '''
    digits = load_digits()
    X = digits.data
    y = digits.target

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=.2, stratify=y)
    return Xtrain, Xtest, ytrain, ytest
