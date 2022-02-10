# 

**Inferential Statistics**

With inferential statistics, we try to reach conclusions that extend beyond the immediate data alone. For instance, we use inferential statistics to try to infer from the sample data what the overall population might think. Or, we use inferential statistics to make judgments of the probability of the overall population. This is also know as Point Estimation.

• Sampling Analysis
• Inferential Statistics
• Sampling Distribution
• Central Limit Theorem
• Central Limit Theorem Exercise

Hypothesis Testing
• Hypothesis and hypothesis Testing
• One tail/Two tail test
• level of Significance & Confidence Interval
• P Value
• Type I and Type II Errors

**Central Limit Theorem (CLT)**
"Sample Mean will be approximately normally distributed for larger sample size regardless of the original distribution from which we are taking samples."
Sampling Mean = Population Mean (μ)
Sampling SD = σ /√n
{in case σ is not known then SD = s/ √n, s = Sample SD}
Application of CLT
From CLT we know, sampling SD σx = σ /√n
From Standard Normal distribution we know – Z = (x - μ) / σ
So for any sampling distribution we can say – Z = (X - μ) / (σ /√n), so now we can calculate the probability using SND for any Non
normal Population.

**Hypothesis**
A hypothesis (plural hypotheses) is a proposed explanation for a phenomenon. In Statistics Hypothesis can be any theory about the data that we want to validate (generally accept or reject) – we will be mainly working of two type of hypotheses :
1. Null Hypothesis (H0) – Current Assumption or Theory which is currently assumed to be correct
2. Alternative Hypothesis (H1) – Claim or theory that we want to prove
Ex. H0: While flipping a coin the probability of getting head is 0.5;
H1 : Probability of getting head is less than 0.5

**Hypothesis Testing**
Validating the null hypothesis (H0) against some Alternative Hypothesis (H1) based on some given sample data. P Value & level of Significance
• P value – Probability of getting the given sample or even more
extreme samples if null hypothesis is true
• Significance level (α ) – Minimum P Value to accept Null
Hypothesis

# dataScience

**Linear_regression_MSE_R2_AdjustedR2.ipynb**
**
There are three main assumptions linear regression makes:**

The independent variables have a linear relationship with the dependent variable.

    .If the independent variables do not have a linear relationship with the dependent variables, there’s no point modeling them using LINEAR regression.
    .MSE should be less
  
The variance of the dependent variable is uniform across all combinations of Xs

    .we need something called homoscedasticity. In simple terms, it means that the residuals (y_predicted - y) must have constant variance.
    .heteroscedasticity (residuals are entirely random. One can hardly see any pattern) is something you want to avoid    

The error term e associated with Y and Y’ is independent and identically distributed.

    .What is autocorrelation? We know that correlation measures the degree of linear relationship between two variables, say A and B. Autocorrelation measures the correlation of a variable with itself. For example, we want to measure how dependent a particular value of A correlates with the value of A some t steps back. Such type of patterns may frequently occur in time series data (where X is time, and Y is a property that varies with time. Stock prices for instance). The unaccounted patterns here could be some seasonality or trends.

**Evaluating a Model**

Previously, we defined MSE to calculate the errors committed by the model. However, if I tell you that for some data and some model the MSE is 23.223. Is this information alone enough to say something about the quality of our fit? How do we know if it’s the best our model can do? We need some benchmark to evaluate our model against. Hence, we have a metric called R squared (R^2).

Let’s get the terms right. We know MSE. However, what is TSE or Total Squared Error? Suppose we had no X. We have Y, and we asked to model a line to fit these Y values such that the MSE minimizes. Since we have no X, our line would be of the form Y’ = a, where a is a constant. If we substitute Y’ for a in the MSE equation, and minimize it by differentiating with respect to a and set equal to zero, it turns out that  a = mean(Y)  gives the least error. Think about this – the line Y’ = a can be understood as the baseline model for our data. Addition of any independent variable X improves our model. Our model cannot be worse than this baseline model. If our X didn’t help to improve the model, it’s weight or coefficients would be 0 during MSE minimization. This baseline model provides a reference point. Now come back to R squared and take a look at the expression. If our model with all the X and all the Y produces an error same as the baseline model (TSE), R squared = 1-1 = 0. This is the worst case. On the opposite, if MSE =0, R squared = 1 which is the best case scenario.

Now let’s take a step back and think about the case when we add more independent variables to our data. How would the model respond to it? Suppose we are trying to predict house prices. If we add the area of the house to our model as an independent variable, our R square could increase. It is obvious. The variable does affect house prices. Suppose we add another independent variable. Something garbage, say random numbers. Can our R square increase? Can it decrease?

Now, if this garbage variable is helping minimize MSE, it’s weight or coefficient is non zero. If it isn’t, the weight is zero. If so, we get back the previous model. We can conclude that adding new independent variable at worst does nothing. It won’t degrade the model R squared. So if I keep adding new variables, I should get a better R squared. And I will. However, it doesn’t make sense. Those features aren’t reliable. Suppose if those set of random numbers were some other set of random numbers, our weights would change. You see, it is all up to chance. Remember that we have a sample of data points on which we build a model. It needs to be robust to new data points out of the sample. That’s why we introduce something called adjusted R squared. Adjusted R squared penalizes any addition of independent variables that do not add a significant improvement to the model. You usually use this metric to compare models after the addition of new features.

n is the number of points, k is the number of independent variables. If you add features without a significant increase in R squared, the adjusted R squared decreases.

So now we know something about linear regression. We dive deeper in the second part of the blog. In the next blog, we look at regularization and assessment of coefficients.



**Logistics regression **
**Evaluating a Model**
for the logistics refression, the evaluation matrix would be the confussion matrix.

Precisson recall sensitivity specificity: used based on industry.
Standardized equations
sensitivity = recall = tp / t = tp / (tp + fn)
specificity = tn / n = tn / (tn + fp)
precision = tp / p = tp / (tp + fp)
Equations explained
Sensitivity/recall – how good a test is at detecting the positives. A test can cheat and maximize this by always returning “positive”.
Specificity – how good a test is at avoiding false alarms. A test can cheat and maximize this by always returning “negative”.
Precision – how many of the positively classified were relevant. A test can cheat and maximize this by only returning positive on one result it’s most confident in.
The cheating is resolved by looking at both relevant metrics instead of just one. E.g. the cheating 100% sensitivity that always says “positive” has 0% specificity.
Accuracy, sensitivity, specificity: plot using cutoff_dff.plot.line(x=prob, y[accu, sensitivity, specificity ]) 
Plot.show



Let’s dig deep into all the parameters shown in the figure above.
The first thing you will see here is ROC curve and we can determine whether our ROC curve is good or not by looking at AUC (Area Under the Curve) and other parameters which are also called as Confusion Metrics. A confusion matrix is a table that is often used to describe the performance of a classification model on a set of test data for which the true values are known. All the measures except AUC can be calculated by using left most four parameters. So, let’s talk about those four parameters first.

True positive and true negatives are the observations that are correctly predicted and therefore shown in green. We want to minimize false positives and false negatives so they are shown in red color. These terms are a bit confusing. So let’s take each term one by one and understand it fully.
True Positives (TP) - These are the correctly predicted positive values which means that the value of actual class is yes and the value of predicted class is also yes. E.g. if actual class value indicates that this passenger survived and predicted class tells you the same thing.
True Negatives (TN) - These are the correctly predicted negative values which means that the value of actual class is no and value of predicted class is also no. E.g. if actual class says this passenger did not survive and predicted class tells you the same thing.
False positives and false negatives, these values occur when your actual class contradicts with the predicted class.
False Positives (FP) – When actual class is no and predicted class is yes. E.g. if actual class says this passenger did not survive but predicted class tells you that this passenger will survive.
False Negatives (FN) – When actual class is yes but predicted class in no. E.g. if actual class value indicates that this passenger survived and predicted class tells you that passenger will die.
Once you understand these four parameters then we can calculate Accuracy, Precision, Recall and F1 score.
Accuracy - Accuracy is the most intuitive performance measure and it is simply a ratio of correctly predicted observation to the total observations. One may think that, if we have high accuracy then our model is best. Yes, accuracy is a great measure but only when you have symmetric datasets where values of false positive and false negatives are almost same. Therefore, you have to look at other parameters to evaluate the performance of your model. For our model, we have got 0.803 which means our model is approx. 80% accurate.
Accuracy = TP+TN/TP+FP+FN+TN
Precision - Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. The question that this metric answer is of all passengers that labeled as survived, how many actually survived? High precision relates to the low false positive rate. We have got 0.788 precision which is pretty good.
Precision = TP/TP+FP
Recall (Sensitivity) - Recall is the ratio of correctly predicted positive observations to the all observations in actual class - yes. The question recall answers is: Of all the passengers that truly survived, how many did we label? We have got recall of 0.631 which is good for this model as it’s above 0.5.
Recall = TP/TP+FN
IMP
F1 score - F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. Intuitively it is not as easy to understand as accuracy, but F1 is usually more useful than accuracy, especially if you have an uneven class distribution. Accuracy works best if false positives and false negatives have similar cost. If the cost of false positives and false negatives are very different, it’s better to look at both Precision and Recall. In our case, F1 score is 0.701.
F1 Score = 2*(Recall * Precision) / (Recall + Precision)
So, whenever you build a model, this article should help you to figure out what these parameters mean and how good your model has performed.
I hope you found this blog useful. Please leave comments or send me an email if you think I missed any important details or if you have any other questions or feedback about this topic.

**Random_Forest’s_hyperparameters.ipynb**

Understanding Random Forest’s hyperparameters with images

About Random Forest


Decision Tree is a disseminated algorithm to solve problems. It tries to simulate the human thinking process by binarizing each step of the decision. So, at each step, the algorithm chooses between True or False to move forward.

That algorithm is simple, yet very powerful, thus widely applied in machine learning models. However, one of the problems with Decision Trees is its difficulty in generalizing a problem. The algorithm learns so well how to decide about a given dataset that when we want to use it to new data, it fails giving us the best answer.

To solve that problem, a new type of Decision Tree algorithm was created by gathering many Trees trained over variations of the same dataset and using a voting or average system to combine them and decide the best result for each data point. That is the concept of Random Forest.

    A random forest is a classifier consisting of a collection of tree structured classifiers (…) independent identically distributed random vectors and each tree casts a unit vote for the most popular class at input x . Leo Breiman, 2001.

Creating a Simple Model

Create a model is fairly simple. As many of you may know, the actual model instance, fit and prediction can be done in just a couple of lines. However, the hard part is usually to prepare the data and to tune the model.

To tweak a model, we must change the hyperparameters from the default values to those that will give us the best results. Our goal here is to better understand what each of the hyperparameters from a Random Forest do in order to be better suited to change them when needed.

Here are the imports and dataset I will be using in this example: wines dataset from sklearn .

# Data

import pandas as pd
from sklearn.datasets import load_wine# Data split
from sklearn.model_selection import train_test_split# Model
from sklearn.ensemble import RandomForestClassifier# Visualize Tree
from sklearn.tree import export_graphviz# Load dataset
df = load_wine()# variables
X = pd.DataFrame(df.data, columns=df.feature_names)

# target

y = df.target# Split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=12)

Here is the model with the default values:

# Instantiate class. 

Using random_state=2 for you to be able to reproduce the same resultrf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=2, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)

Let’s train it and generate a picture.

# Fit the model

rf.fit(X_train,y_train)# Extract one of the trees from the model
tree = rf.estimators_[99]# Export as dot file
export_graphviz(tree, out_file='tree.dot',
feature_names = df.feature_names, class_names = df.target_names, rounded = True, proportion = False, precision = 2, filled = True)# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=90'])# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.png')

Picture1: Random Forest model with all the default values. Image by the author.

Ok, great. Now let’s align the names we’re using, before moving on.

    Node is when we have a split.
    Branch is a decision path [e.g. alcohol=True > hue=True > end (leaf)]
    Leaf is the last square of the branch.

So, that said, the first interesting thing to notice in the picture 1 is that the branch only gets to the leaf when the “gini” indicator values 0.0. That indicator is a function to measure the quality of a split. sklearn supports “gini” or “entropy” for the information gain. When the number reaches 0, we can’t gain anymore information, since the leaf is now pure. A pure leaf node is confirmed when you look at the values [class 0 qty, class 1 qty, class2 qty]. If the number in the class predicted is the only one above 0, then the leaf is pure (e.g [0,1,0] and class_1 = pure node).

Well, our goal here is to better understand how changing a hyperparameter can change the quantity of leaves, nodes and branches. So I will change the main hyperparameters one at a time and plot the result.
Max Features

The number of features to consider when looking for the best split. The default value is ‘auto’, which uses ‘sqrt’, but it has other options, like ‘log2’ or an interesting possibility to enter a float number between 0 and 1 and that will be the percentage of features used at each split. If you have 10 features and use max_feature=0.2 , it will consider 20% of the features, which is 2.

Usually not all features are that important, so this is a good hyperparameter to test in a GridSearchCV and you can try starting with values like 0.3, 0.4. The smaller the number here, the smaller the variance, but higher bias. For higher numbers, you have more chance to have the best features used for split, thus you will reduce the bias, but increase variance.
Picture 2: max_features comparison. Image by the author
Max Depth

This hyperparameter will limit the maximum quantity of splits that the tree can grow down.

# Instantiate class
rf = RandomForestClassifier(max_depth = 2)

Picture 3: RF model with max_depth=2

So, as we have chosen max_depth=2 , it means it can only split two times, making the resulting squares on the 3rd row from Picture 3 as leaves. Notice that the gini indicator is pure for only one square. In fact, it does not have too much influence in the result as there are 100 different trees (estimators) in this model. Even with the depth limited to 1, it still predicted the three classes. It must be used together with other hyperparameters.
Minimum Samples for Split

The minimum of samples in a given node to be able to split.

Look again at Picture 3. See that we have 42 samples on the left side. Let’s set our min_samples_split at 50 and see what happens.

rf = RandomForestClassifier(min_samples_split=50, random_state=2)


As expected, the left branch did not grow. Therefore, that is another way to prune a tree and force it to give a classification prior to reach the node purity.
Maximum Leaf Nodes

It determines the maximum leaves you will have in your tree.

rf = RandomForestClassifier(max_leaf_nodes=4, random_state=2)

Here, we see that the classifier created one leaf for each predictable class (class 0, 1 or 2).
Minimum Samples per Leaf

The number of samples that the leaf needs to have. It means that if the number of leaves will be below that amount after another split, it won’t be processed.

rf = RandomForestClassifier(min_samples_leaf=20, random_state=2)


See in Picture 5 that the number of samples in each leaf is higher than 20. When our model is overfitting, we can try to tweak this hyperparameter combined or not with max_depth and force an earlier decision, what may help to generalize predictions.
Complexity Cost Pruning

Another way to prune a tree is using the ccp_alpha hyperparameter, which is the complexity cost parameter. The algorithm will choose between trees by calculating the complexity cost and the amounts with smaller values are considered weaker, so they are pruned. The pruning stops once the smallest value of the complexity cost is higher than the ccp_alpha. Greater values of ccp_alpha increase the number of nodes pruned.
Picture 6: Differences in ccp_alpha numbers. Image by the author.
Before You Go

The intent of this material was to give a visual idea of how changing each of the hyperparameters from a Random Forest model will affect your result.

    n_estimators: number of estimators. The more you have you should have a more accurate result, but it is more expensive in terms of computational power.
    criterion: choose between gini or entropy. Both will seek the same result, that is node purity.
    max_depth: the larger a tree is, the more chance of overfitting it has. RF models usually try to minimize that, but this hyperparameter can be an interesting one to play if your model is overfitting.
    min_samples_split : this one work together with the one above. That is the minimum samples needed to split to another branch.
    max_leaf_nodes : can force the tree to have less leaves.
    ccp_alpha: another way to prune the tree, based on calculations of the complexity cost.
Understanding the effect of the hyperparameters in a Random Forest ML model
