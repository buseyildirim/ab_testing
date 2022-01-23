import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, levene,ttest_ind

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)

    print("##################### Types #####################")
    print(dataframe.dtypes)

    print("##################### Head #####################")
    print(dataframe.head(head))

    print("##################### Tail #####################")
    print(dataframe.tail(head))

    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Max binding
df_control_group = pd.read_excel("datasets/ab_testing.xlsx", sheet_name="Control Group")
# Average binding
df_test_group = pd.read_excel("datasets/ab_testing.xlsx", sheet_name="Test Group")

# Lets analysis control group.
check_df(df_control_group)

#Lets analysis test group.

check_df(df_test_group)

"""The success criterion for Bombomba.com is the Number of Purchases. 
Therefore, we will examine the purchasing variable."""

""" There are no missing values in the two data. Let's check out the outlier."""

A = pd.DataFrame(df_control_group["Purchase"])
check_outlier(A,"Purchase")

sns.boxplot(x=A["Purchase"])
plt.show()

B = pd.DataFrame(df_test_group["Purchase"])
check_outlier(B,"Purchase")

sns.boxplot(x=B["Purchase"])
plt.show()

"There were no outliers in the Purchase variable for either group."

"""Let's start A/B testing."""

print(" Mean of purchase of control group: %.3f" %A.mean(), "\n",
      "Mean of purchase of test group: %.3f" %B.mean())

"""There is a mathematical difference when looking at the purchasing rates for the two groups.
 Group B, the average binding strategy, seems to be more successful. 
 But we do not know whether this difference is statistically significant. 
 To understand this, we must apply hypothesis testing."""

################################################ ####
# AB Testing (Independent Two-Sample T-Test)
################################################ ####

# It is used when it is desired to make a comparison between the mean of two groups.

#1. Assumption Check
# - 1. Normality Assumption
# - 2. Variance Homogeneity
# 2. Implementation of the Hypothesis
# - 1. Independent two-sample t-test (parametric test) if assumptions are met.
# - 2. Mannwhitneyu test if assumptions are not met (non-parametric test).
# Note:
# - If the normality is not provided, the mannwhitneyu test should be applied directly.
# If the normality assumption is provided but the variance homogeneity is not provided,
# the equal_var parameter can be set to False for the two-sample t-test.

"""Hypotheses
# The null hypothesis (H0) and alternative hypothesis (H1).

H0: M1 = M2 (There is no statistical difference between the average purchase earned 
                by the maximum binding strategy and the average purchase achieved by 
                the average binding strategy.)
H1: M1 != M2 (There is a statistical difference between the average purchase
                earned by the maximum binding strategy and the average purchases
                 earned by the average binding strategy.)"""

# Assumption Check

# Normality Assumption

"""H0: There is no statistically significant difference between sample distribution
        and theoretical normal distribution
H1: There is statistically significant difference between sample distribution and 
        theoretical normal distribution
The test rejects the hypothesis of normality when the p-value is less than or equal to 0.05.
We do not want to reject the null hypothesis in the tests that might be considered for assumptions. 

p-value < 0.05 (H0 rejected)
p-value > 0.05 (H0 not rejected)"""

# Shapiro-Wilks Test  for Control Group
test_statistic , pvalue = shapiro(A)
print('Test statistic = %.4f, p-Value = %.4f' % (test_statistic, pvalue))
"""
p-value = 0.5891.
Since the p value is not less than 0.05, the h0 hypothesis cannot be rejected.
That is, the assumption of normality is provided for the control group.
"""

# Shapiro-Wilks Test  for Control Group
test_statistic_b , pvalue = shapiro(B)
print('Test statistic = %.4f, p-Value = %.4f' % (test_statistic, pvalue))
"""
p-value = 0.1541.
Since the p value is not less than 0.05, the h0 hypothesis cannot be rejected.
That is, the assumption of normality is provided for the control group.
"""


# Variance Assumption

"""
H0: the compared groups have equal variance.
H1: the compared groups do not have equal variance.
"""

# Levene Test

test_statistic, pvalue = levene(A["Purchase"],B["Purchase"])
print('Test statistic = %.4f, p-Value = %.4f' % (test_statistic, pvalue))

"""
p-value=0.1083
Since the p value is not less than 0.05, the h0 hypothesis cannot be rejected.
That is, the assumption of variance is provided.
"""

test_stat, pvalue = ttest_ind(A["Purchase"], B["Purchase"], equal_var=True)
print('tvalue = %.4f, pvalue = %.4f' % (test_stat, pvalue))

"""
p-value=0.3493
Since the p value is not less than 0.05, the h0 hypothesis cannot be rejected.
So, There is no statistically significant difference between the Control( “maximum bidding”) campaign
and Test group(average bidding) campaign.
"""