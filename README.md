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

