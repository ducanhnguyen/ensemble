# ensemble
Ensemble methods (randomforest, bagging, adaboost, etc.)

### The problem of decision tree

Decision tree is prone to overfitting (i.e., low bias, high variance). The model created by decision tree is sensitive to dataset. Only a small change in dataset will make a huge change in the model.

In order to solve this problem, we can apply ensemble methods. I study on three main ensemble methods including Bootstrap aggregating (bagging), random forest, and adaboost.

### Bootstrap aggregating (bagging)

Invented in 1994. Widely used in decision tree.

Bagging works well with strong and complex model. The idea is very simple: 

- Step 1: Create B models for B training dataset which are drawn from the original dataset. Each model may be overfitting with its specific dataset (i.e., low bias, high variance)

- Step 2: Combine these B models may make the model more generally. As a result, the combined model will keep the same low bias while lowering variance.

- Step 3: Make prediction: compute average if regression, voting if classification

### RandomForest

Invented in 1995. Only used in decision tree.

Iead: rather building a model on the whole features (D features), we only select a subset of these features (d features). The author suggested as follows:

- Classification: d = floor(sqrt(D)) (at least 1 feature)

- Regression: d = floor(D/3) (at least 5 features)

### Adaboost

Work well with weak models. 

Idea: Adaboost will make a linear combination of weak models to make a strong model.

|Column 1|Column 2|
| --- | --- |
|<img src="https://github.com/ducanhnguyen/ensemble/blob/master/ensemble/img/decision_tree.png" width="450">|<img src="https://github.com/ducanhnguyen/ensemble/blob/master/ensemble/img/decision_tree_adaboost.png" width="450">|
|<img src="https://github.com/ducanhnguyen/ensemble/blob/master/ensemble/img/decision_tree_rf.png" width="450">|<img src="https://github.com/ducanhnguyen/ensemble/blob/master/ensemble/img/decision_tree_bagging.png" width="450">|
