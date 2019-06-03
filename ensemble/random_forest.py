'''
This example demonstrates the predictive power of random forest in comparison with no-random forest.
Decision tree is prone to overfitting.
Random forest helps to reduce the variance of decision tree while still keeping low bias.
'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

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
    plt.title('Decision tree (no random forest)')
    plt.show()

    print('Highest accuracy when not using random forest = %f' % np.max(accuracies))


def decision_tree_random_forest(Xtrain, Xtest, ytrain, ytest, ensemble_size=60):
    # random forest
    accuracies = []
    ensemble_sizes = []

    D = Xtrain.shape[1]

    for i in range(1, ensemble_size):
        rf = RandomForestClassifier(
            n_estimators=i,
            bootstrap=True,
            max_features=int(np.floor(np.sqrt(D))))

        rf.fit(Xtrain, ytrain)

        ypred = rf.predict(Xtest)
        accuracy = np.mean(ypred == ytest)

        ensemble_sizes.append(i)
        accuracies.append(accuracy)

    plt.plot(ensemble_sizes, accuracies)
    plt.xlabel('number of estimators')
    plt.ylabel('accuracy')
    plt.grid(True)
    plt.title('Decision tree (random forest)')
    plt.show()

    print('Highest accuracy of random forest = %f' % np.max(accuracies))


Xtrain, Xtest, ytrain, ytest = ensemble.utils.load_data()
decision_tree(Xtrain, Xtest, ytrain, ytest)
decision_tree_random_forest(Xtrain, Xtest, ytrain, ytest)
