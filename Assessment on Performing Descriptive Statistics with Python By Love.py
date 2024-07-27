#!/usr/bin/env python
# coding: utf-8

# # Assessment on Performing Descriptive Statistics with Python

# In[229]:


# Importing Necessary Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gmean
from scipy.stats import zscore
from scipy.stats import linregress


# # Section 1

# 1. A statistics test was conducted for 10 learners in a class. The mean of their score is 85 and the variance of the score is zero. What can you interpret about the score obtained by all learners?
Interpretation:

# Mean (Average) Score: The mean score of the 10 learners is 85. This represents the central tendency of their scores.
# Variance: The variance of the score is zero, there is no variability in the scores. 
# All learners achieved the same score is 85.
# 2. In a residential locality, the mean size of the house is 2224 square feet and the median value of the house is 1500 square feet. What can you interpret about the skewness in the distribution of house size? Are there bigger or smaller houses in the residential locality?
Interpretation:

# Mean Size of Houses: The mean size of houses in the residential locality is 2224 square feet.
# Median Size of Houses: The median size of houses in the residential locality is 1500 square feet.

# If the mean exceeds the median (Mean > Median), the distribution tends to be positively skewed (right-skewed). 
# There are more smaller houses in the residential locality, but there are also some significantly larger houses that are increasing the mean size of the houses.
# 3.The following table shows the mean and variance of the expenditure for two groups of people.You want to compare the variability in expenditure for both groups with respect to their mean. Which statistical measure would you use to evaluate the variability in expenditure? Please provide an explanation for your answer.
# To compare the variability in expenditure for both groups with respect to their mean, you would use the Coefficient of Variation (CV).
# For each group, calculate the CV using the formula: CV = (Standard Deviation / Mean) * 100%
# In[2]:


# Calculate CV for both groups

cv1 = (125000 / 500000) * 100
cv2 = (10000 / 40000) * 100

print("Group 1 CV:", int(cv1),"%")
print("Group 2 CV:", int(cv2),"%")

# Conclusion: Both groups have the same CV of 25%. This implies that the relative variability in expenditure compared to the mean is the same for both groups.
# 4. During the survey, the ages of 80 patients infected by COVID and admitted to one of the city hospitals were recorded and the collected data is represented in the less than cumulative frequency distribution table.

# In[5]:


# Create data

age = ['5-15', '15-25', '25-35', '35-45', '45-55', '55-65']
num_patients = [6, 11, 21, 23, 14, 5]


# a. Which class interval has the highest frequency?

# In[7]:


highest_freq = max(num_patients)
highest_freq_index = num_patients.index(highest_freq)

print(f"The class interval with the highest frequency is: {age[highest_freq_index]} with a frequency of {highest_freq}")


# b. Which age was affected the least?

# In[8]:


least = min(num_patients)
least_index = num_patients.index(least)

print(f"The age interval affected the least is: {age[least_index]} with a frequency of {least}")


# c. How many patients aged 45 years and above were admitted?

# In[9]:


# Sum the frequencies of the 45 years and above aged patients

total_patients_45_and_above = sum(num_patients[4:])

print(f"Total patients aged 45 years and above: {total_patients_45_and_above}")


# d. Which is the modal class interval in the above dataset

# In[11]:


highest_freq = max(num_patients)
highest_freq_index = num_patients.index(highest_freq)

print(f"The modal class interval is: {age_intervals[max_freq_index]} with a frequency of {max_freq}")


# e. What is the median class interval of age?

# In[41]:


# Create a pandas DataFrame

df = pd.DataFrame({'Age': age, 'Num Patients': num_patients})
print(df)


# In[42]:


# Sort the DataFrame by Num Patients

sort = df['Num Patients'].sort_values
print(sort)


# In[43]:


# Calculate the cumulative sum

Cum_Sum = df['Num Patients'].cumsum()
print(Cum_Sum)


# In[44]:


# Find the median index

median = df[Cumulative_Sum >= df['Num Patients'].sum() / 2].index[0]

median_age = df.loc[median, 'Age']

print(f'The median class interval of age is {median_age}.')


# 5. Assume you are the trader and you have invested over the years, and you are worried about the average return on investment. What average method would you use to compute the average return for the data given below?

# In[223]:


# Create a DataFrame from the data

data = {'year': [2015, 2016, 2017, 2018, 2019, 2020],
        'return': [0.36, 0.23, -0.48, -0.30, 0.15, 0.31],
        'asset_price': [5000, 6400, 7890, 9023, 4567, 3890]}

df = pd.DataFrame(data)
print(df)


# In[224]:


# Calculate the geometric mean return

geometric_mean_return = gmean(1 + df['return']) - 1

print(f"Geometric Mean Return: {geometric_mean_return:.2%}")


# 6. Suppose you have been told to measure the average height of all the males on the earth. What would be your strategy for the same? Would the average height be a parameter or a statistic? Justify your answer.
- A parameter is a characteristic of the entire population, which is often unknown.
- A statistic is a numerical value calculated from a sample of the population, used to estimate the parameter.

In this case, the average height calculated from the sample data is a statistic because it's an estimate of the true average height of all males on Earth (the parameter). The statistic is subject to sampling error and variability, whereas the parameter is a fixed value representing the entire population.
# In[76]:


# Create a DataFrame from Sample dataset

data = { 'Height (cm)': [175, 180, 165, 178, 185, 170, 179, 176, 181, 182]}

df = pd.DataFrame(data)
print(df)


# In[78]:


# Calculate the average height

average_height = df['Height (cm)'].mean()

print(f"Average height: {average_height} cm")


# 7. Calculate the z score of the following numbers:
# X = [4.5,6.2,7.3,9.1,10.4,11]

# In[68]:


# Define the dataset

x = [4.5, 6.2, 7.3, 9.1, 10.4, 11]
print(x)


# In[69]:


# Calculate the mean

mean_x = np.mean(x)
print(f"Mean: {mean_x:.2f}")


# In[70]:


# Calculate the standard deviation

std_x = np.std(x)
print(f"Standard Deviation: {std_x:.2f}")


# In[71]:


# Calculate the z-scores

z_scores = [(x - mean_x) / std_x for x in x]
print("Z-scores:", z_scores)


# # Section 2

# You are expected to perform statistical analysis for the Bank Personal Loan Modelling dataset. Below is the data dictionary. For questions, 8 to 20 use the Bank Personal Loan Modelling dataset and answer the given questions.

# In[79]:


bank = pd.read_csv(r"D:\Bank Personal Loan Modelling.csv")
bank


# In[80]:


bank.info()


# 8. Give us the statistical summary for all the variables in the dataset.

# In[81]:


bank.describe().T


# 9. Evaluate the measures of central tendency and measures of dispersion for all the quantitative variables in the dataset.
All Quantitative variables in the dataset
# In[82]:


b = bank.select_dtypes(int)
b


# Measures of Central Tendency

# In[83]:


b.mean()


# In[84]:


b.median()


# Mode - (Categorical data & Discrete data) From All Quantitative variables in the dataset

# In[85]:


b['Family'].value_counts()


# In[86]:


b['Family'].mode()


# In[87]:


b['Education'].value_counts()


# In[88]:


b['Education'].mode()


# Measures of Dispersion

# In[89]:


# Range

b.max()-b.min()


# In[90]:


# IQR

b.quantile(0.75) - b.quantile(0.25)


# In[91]:


# Variance

b.var()


# In[92]:


# Standard Deviation

b.std()


# In[93]:


b.describe().T


# 10. What statistical method will you use to examine the presence of a linear relationship between age and experience variables? Also, create a plot to illustrate this relationship.

# In[96]:


# linear relationship between age and experience variables with the help of correlation

df = bank.iloc[:,[1,2]]
df.corr()


# In[227]:


# Calculate the correlation coefficient

correlation = bank['Age'].corr(bank['Experience'])

print(f"Correlation coefficient: {correlation:.2f}")


# In[230]:


# Perform simple linear regression

slope, intercept, r_value, p_value, std_err = linregress(bank['Age'], bank['Experience'])
print(f"Slope: {slope:.4f}, Intercept: {intercept:.4f}")


# In[231]:


# Create scatter plot with regression line

plt.figure(figsize=(10, 6))
plt.scatter(bank['Age'], bank['Experience'], color='blue', label='Data points')
plt.plot(bank['Age'], intercept + slope * bank['Age'], color='red', label='Regression line')
plt.xlabel('Age')
plt.ylabel('Experience')
plt.title('Age vs. Experience with Linear Regression')
plt.legend()
plt.show()


# 11. What is the most frequent family size observed in this dataset?

# In[135]:


# Calculate the frequency of each family size

family_size = bank['Family'].value_counts()
print(family_size)


# In[140]:


# Print the most frequent family size

print("Most frequent family size:", family_size.index[0], "with a frequency of", family_size.iloc[0])


# 12. What is the percentage of variation you can observe in the ‘Income’ variable?

# In[142]:


# Calculate the mean and standard deviation of the 'Income' variable

mean_income = bank['Income'].mean()
std_income = bank['Income'].std()


# In[144]:


# Calculate the percentage of variation

percent_variation = (std_income / mean_income) * 100

print(f"Percentage of variation in 'Income': {percent_variation:.2f}%")


# 13. The ‘Mortgage’ variable has a lot of zeroes. Impute with some business logical value that you feel fit for the data.

# In[214]:


# Calculate the median Mortgage for non-zero values

median_mortgage = bank[bank['Mortgage'] > 0]['Mortgage'].median()
print("Median Mortgage for non-zero values:", median_mortgage)


# In[220]:


# Impute missing values with the median Mortgage

bank['Mortgage'] = bank['Mortgage'].replace(0, median_mortgage)

print("Imputed Mortgage values:")
print(bank['Mortgage'])


# 14. Plot a density curve of the CCAvg variable for the customers who possess credit cards and write an interpretation about its distribution.

# In[155]:


# Filter the DataFrame to include only customers with credit cards

credit_cards = bank[bank['CreditCard'] == 1]


# In[156]:


# Plot a density curve of the CCAvg variable for the customers who possess credit cards

sns.kdeplot(credit_cards['CCAvg'], label='CCAvg (with credit cards)')
plt.title('Density Curve of CCAvg for Customers with Credit Cards')
plt.xlabel('Average Credit Card Balance')
plt.ylabel('Density')
plt.show()

# Interpretation about its distribution :-

The density curve of the CCAvg variable for customers with credit cards shows a right-skewed distribution.
This indicates that most customers with credit cards have relatively low average credit card balances, while a smaller number of customers have much higher balances.
# 15. Do you see any outliers in the dataset? If yes, what plot you would think will be suitable to showcase to the stakeholders?

# In[158]:


# Calculate the interquartile range (IQR)

Q1 = bank['CCAvg'].quantile(0.25)
Q3 = bank['CCAvg'].quantile(0.75)
IQR = Q3 - Q1


# In[160]:


# Identify outliers using the IQR method

outliers = bank[(bank['CCAvg'] < (Q1 - 1.5 * IQR)) | (bank['CCAvg'] > (Q3 + 1.5 * IQR))]


# In[168]:


# Create a boxplot to showcase outliers
sns.boxplot(x='CCAvg', data=bank)
plt.title('Boxplot of CCAvg with Outlier')
plt.show()


# 16. Give us the decile values of the variable ‘Income’ in the dataset.

# In[175]:


# Calculate deciles for the 'Income' column

deciles = bank['Income'].quantile([0.1 * i for i in range(1, 10)])

print("Decile values for 'Income':")
print(deciles)


# 17. Give the IQR of all the variables which are quantitative and continuous.

# In[179]:


# Select quantitative and continuous variables

quant_vars = bank.select_dtypes(include=['int64', 'float64'])


# In[180]:


# Calculate IQR for each variable

iqr = quant_vars.quantile(0.75) - quant_vars.quantile(0.25)
print(iqr)


# 18. Do the higher-income holders spend more on credit cards?

# In[204]:


# Calculate the median income

median_income = bank['Income'].median()


# In[206]:


# Split the data into higher and lower income groups

higher_income = bank[bank['Income'] > median_income]
lower_income = bank[bank['Income'] <= median_income]


# In[210]:


# Calculate the median CCAvg for each group

higher_income_ccavg = higher_income['CCAvg'].median()
lower_income_ccavg = lower_income['CCAvg'].median()

print("Median CCAvg for higher-income holders:", higher_income_ccavg)
print("Median CCAvg for lower-income holders:", lower_income_ccavg)


# In[211]:


# Check if higher-income holders spend more on credit cards

if higher_income_ccavg > lower_income_ccavg:
    print("Higher-income holders spend more on credit cards.")
else:
    print("Higher-income holders do not spend more on credit cards.")


# 19. How many customers use online banking? Do customers using bank internet facilities have higher incomes?

# In[187]:


# Calculate the number of customers using online banking

online_bankers = bank['Online'].sum()
print("Number of customers using online banking:", online_bankers)


# In[188]:


# Calculate the average income for online higher_income_avg and lower_income_avg

higher_income_avg = bank[bank['Online'] == 1]['Income'].median()
lower_income_avg = bank[bank['Online'] == 0]['Income'].median()

print("Average income for online bankers:", online_income_avg)
print("Average income for non-online bankers:", non_online_income_avg)


# In[189]:


# Check if online banking users have higher incomes

if online_income_avg > non_online_income_avg:
    print("Customers using bank internet facilities have higher incomes.")
else:
    print("Customers using bank internet facilities do not have higher incomes.")


# 20. Using the z-score of the income variable, find out the number of observations outside the +-3σ.

# In[196]:


# Calculate the z-scores for each observation

bank['Income_zscore'] = zscore(bank['Income'])


# In[202]:


# Find observations outside +3σ

outliers = bank[(bank['Income_zscore']) > 3]

print("Number of observations outside +3σ:", len(outliers))
print("Outliers:")
print(outliers)


# In[221]:


# Find observations outside -3σ

outliers = bank[(bank['Income_zscore']) < -3]

print("Number of observations outside -3σ:", len(outliers))
print("Outliers:")
print(outliers)


# In[ ]:




