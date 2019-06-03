'''
This example demonstrates the predictive power of bagging in comparison with no-bagging.
Decision tree is prone to overfitting.
Bagging helps to reduce the variance of decision tree while still keeping low bias.
'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.ensemble.bagging import BaggingClassifier

import ensemble.utils


def decision_tree(Xtrain, Xtest, ytrain, ytest, max_depth=30):
    # no bagging
    max_depths = []
    accuracies = []

    for i in range(1, max_depth):
        decision_tree = tree.DecisionTreeClassifier(max_depth=i)
        decision_tree.fit(Xtrain, ytrain)
        ypred = decision_tree.predict(Xtest)

        accuracy = np.mean(ypred == ytest)
        accuracies.append(accuracy)

        max_depths.append(i)

    plt.plot(max_depths, accuracies)
    plt.xlabel('max depth')
    plt.ylabel('accuracy')
    plt.grid(True)
    plt.title('Decision tree (no ensemble)')
    plt.show()

    print('Highest accuracy when not using bagging = %f' % np.max(accuracies))


def decision_tree_bagging(Xtrain, Xtest, ytrain, ytest, ensemble_size=60):
    # bagging
    accuracies = []
    ensemble_sizes = []

    for i in range(1, ensemble_size):
        bagging = BaggingClassifier(
            base_estimator=tree.DecisionTreeClassifier(),
            n_estimators=i,
            bootstrap=True,
            max_samples=1.0,
            max_features=1.0)

        bagging.fit(Xtrain, ytrain)

        ypred = bagging.predict(Xtest)
        accuracy = np.mean(ypred == ytest)

        ensemble_sizes.append(i)
        accuracies.append(accuracy)

    plt.plot(ensemble_sizes, accuracies)
    plt.xlabel('number of estimators')
    plt.ylabel('accuracy')
    plt.grid(True)
    plt.title('Decision tree (bagging)')
    plt.show()

    print('Highest accuracy of  bagging = %f' % np.max(accuracies))


Xtrain, Xtest, ytrain, ytest = ensemble.utils.load_data()
decision_tree(Xtrain, Xtest, ytrain, ytest)
decision_tree_bagging(Xtrain, Xtest, ytrain, ytest)
