# dataScience

**Linear_regression_MSE_R2_AdjustedR2.ipynb**

There are three main assumptions linear regression makes:

The independent variables have a linear relationship with the dependent variable.

    .If the independent variables do not have a linear relationship with the dependent variables, there’s no point modeling them using LINEAR regression.
    .MSE should be less
  
The variance of the dependent variable is uniform across all combinations of Xs

    .we need something called homoscedasticity. In simple terms, it means that the residuals (y_predicted - y) must have constant variance.
    .heteroscedasticity (residuals are entirely random. One can hardly see any pattern) is something you want to avoid    

The error term e associated with Y and Y’ is independent and identically distributed.

    .What is autocorrelation? We know that correlation measures the degree of linear relationship between two variables, say A and B. Autocorrelation measures the correlation of a variable with itself. For example, we want to measure how dependent a particular value of A correlates with the value of A some t steps back. Such type of patterns may frequently occur in time series data (where X is time, and Y is a property that varies with time. Stock prices for instance). The unaccounted patterns here could be some seasonality or trends.
