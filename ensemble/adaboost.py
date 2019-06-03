'''
This example demonstrates the predictive power of adaboost in comparison with no-adaboost.
Decision tree is prone to overfitting.
Adaboost helps to reduce the variance of decision tree while still keeping low bias.
'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier

import ensemble.utils


def decision_tree(Xtrain, Xtest, ytrain, ytest, max_depth=30):
    # no adaboost
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
    plt.title('Decision tree (no adaboost)')
    plt.show()

    print('Highest accuracy when not using adaboost = %f' % np.max(accuracies))


def decision_tree_adaboost(Xtrain, Xtest, ytrain, ytest, depths=100):
    # adaboost
    accuracies = []
    max_depths = []

    for i in range(1, depths):
        rf = AdaBoostClassifier(
            # weak models
            base_estimator=tree.DecisionTreeClassifier(max_depth=i),
            n_estimators=i,
            algorithm='SAMME.R')

        rf.fit(Xtrain, ytrain)

        ypred = rf.predict(Xtest)
        accuracy = np.mean(ypred == ytest)

        max_depths.append(i)
        accuracies.append(accuracy)

    plt.plot(max_depths, accuracies)
    plt.xlabel('depth')
    plt.ylabel('accuracy')
    plt.grid(True)
    plt.title('Decision tree (adaboost)')
    plt.show()

    print('Highest accuracy of adaboost = %f' % np.max(accuracies))


Xtrain, Xtest, ytrain, ytest = ensemble.utils.load_data()
decision_tree(Xtrain, Xtest, ytrain, ytest)
decision_tree_adaboost(Xtrain, Xtest, ytrain, ytest)
