# ensemble
Ensemble methods (randomforest, bagging, adaboost, etc.)

### The problem of decision tree

Decision tree is prone to overfitting (i.e., low bias, high variance). The model created by decision tree is sensitive to the dataset. Only a small change in the dataset will make a huge change in the model.

In order to solve this problem, we can apply ensemble methods. I study on three main ensemble methods including bootstrap aggregating (bagging), random forest, and adaboost. Ensemble methods are classified as meta-estimator. 'Meta' means 'more than one model'. 'Estimator' is corresonding to 'model'. 'Meta-estimator' is a combination of single models to make a stronger model.

### Bootstrap aggregating (bagging)

Invented in 1994. Widely used in decision tree.

Bagging works well with strong and complex models (or estimators as well). I used model rather than estimator for simplicity. The idea is very simple: 

- Step 1: Create B models for B training dataset which are drawn from the original dataset. Each model may be overfitting with its specific dataset (i.e., low bias, high variance)

- Step 2: Combine these B models may make a model which more generally. As a result, the combined model will keep the low bias while lowering the variance.

- Step 3: Make a prediction: compute average if regression, voting if classification

### RandomForest

Invented in 1995. Only used in decision tree.

Idea: rather building a model on the whole features (D features), we only select a subset of these features (d features). The author suggested the selection of subset as follows:

- Classification: d = floor(sqrt(D)) (at least 1 feature)

- Regression: d = floor(D/3) (at least 5 features)

### Adaboost

Idea: Adaboost will make a linear combination of weak models to make a strong model. All single models are weak (50-60% accuracy), but its combined model is stronger.

During this process, each sample has its own weight and will be updated many times. We initialize all weights of each sample equally.

Each single model has its own weight and this weight will not be changed once the weight is initialized.

We only add one model at a time (called additive modeling) by training on all data.

### Experiments

As it can be seen, ensemble methods outperform the original decision tree. All of them overcome 90% accuracy easily while decision tree cannot reach this high accuracy.

|Column 1|Column 2|
| --- | --- |
|<img src="https://github.com/ducanhnguyen/ensemble/blob/master/ensemble/img/decision_tree.png" width="450">|<img src="https://github.com/ducanhnguyen/ensemble/blob/master/ensemble/img/decision_tree_adaboost.png" width="450">|
|<img src="https://github.com/ducanhnguyen/ensemble/blob/master/ensemble/img/decision_tree_rf.png" width="450">|<img src="https://github.com/ducanhnguyen/ensemble/blob/master/ensemble/img/decision_tree_bagging.png" width="450">|
