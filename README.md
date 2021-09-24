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
