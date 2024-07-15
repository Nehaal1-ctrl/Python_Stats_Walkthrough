#!/usr/bin/env python
# coding: utf-8

# ### 3.1. Statistics in Python

# #3.1.1.1. Data as a table

# In[1]:


#reading the csv file
import pandas
data = pandas.read_csv('brain_size.csv', sep=';', na_values=".")
data  


# In[2]:


#creating from arrays
import numpy as np
t = np.linspace(-6, 6, 20)
sin_t = np.sin(t)
cos_t = np.cos(t)


# In[3]:


pandas.DataFrame({'t': t, 'sin': sin_t, 'cos': cos_t})  


# In[4]:


data.shape    # 40 rows and 8 columns
data.columns  # It has columns  
print(data['Gender']) 
# Simpler selector; looking at the total males and females in the study
data[data['Gender'] == 'Female']['VIQ'].mean()


# In[5]:


#looking at Female average VIQ
groupby_gender = data.groupby('Gender')
for gender, value in groupby_gender['VIQ']:
    print((gender, value.mean()))
#the mean VIQ for the pupulation both males and females
groupby_gender.mean()


# In[6]:


#the mean VIQ for the pupulation both males and females
groupby_gender.mean()


# In[7]:


#using pandas to use matplotlib to see dataframe stats and scatter matrixs'
from pandas import plotting
plotting.scatter_matrix(data[['Weight', 'Height', 'MRI_Count']])
plotting.scatter_matrix(data[['PIQ', 'VIQ', 'FSIQ']])   


# ### 3.1.2. Hypothesis testing: comparing two groups

# In[8]:


#scipy used for simple statistical tests
from scipy import stats


# ### 3.1.2.1. Student’s t-test: the simplest statistical test

# In[9]:


#testing to see wether the VIQ measure of the population mean is 0
stats.ttest_1samp(data['VIQ'], 0)   
#not 0


# In[10]:


#2-sample t-test: testing for difference across populations
#testing wether VIQ of males versus females is significantlly different
female_viq = data[data['Gender'] == 'Female']['VIQ']
male_viq = data[data['Gender'] == 'Male']['VIQ']
stats.ttest_ind(female_viq, male_viq)   


# ### 3.1.2.2. Paired tests: repeated measurements on the same individuals

# In[11]:


#testing if FISQ and PIQ are significantly different
stats.ttest_ind(data['FSIQ'], data['PIQ'])   


# In[12]:


#using a “paired test”, or “repeated measures test”, because  FSIQ and PIQ are measured on the same individuals so variance due to inter-subject variability is confounding
stats.ttest_rel(data['FSIQ'], data['PIQ'])   


# In[13]:


# 1-sample test on the difference
stats.ttest_1samp(data['FSIQ'] - data['PIQ'], 0)   


# In[14]:


#Wilcoxon signed-rank test to relax Gaussian errors
stats.wilcoxon(data['FSIQ'], data['PIQ'])   


# ### 3.1.3. Linear models, multiple factors, and analysis of variance

# In[15]:


#generating stimulated data
import numpy as np
x = np.linspace(-5, 5, 20)
np.random.seed(1)
# normal distributed noise
y = -5 + 3*x + 4 * np.random.normal(size=x.shape)
# Create a data frame containing all the relevant variables
data = pandas.DataFrame({'x': x, 'y': y})


# In[16]:


#fitting OLS model and viewing the stats
from statsmodels.formula.api import ols
model = ols("y ~ x", data).fit()
print(model.summary())  


# In[17]:


#comparing groups or multiple categories using brain size data
data = pandas.read_csv('brain_size.csv', sep=';', na_values=".")


# In[18]:


#linear model of IQ male vs female 
model = ols("VIQ ~ Gender + 1", data).fit()
print(model.summary())  


# In[19]:


#forcing a model to treat a cetegoricial variable as an integer
model = ols('VIQ ~ C(Gender)', data).fit()


# In[20]:


#comparing different types of IQ as categorical variable
data_fisq = pandas.DataFrame({'iq': data['FSIQ'], 'type': 'fsiq'})
data_piq = pandas.DataFrame({'iq': data['PIQ'], 'type': 'piq'})
data_long = pandas.concat((data_fisq, data_piq))
print(data_long)  
#fitting model
model = ols("iq ~ type", data_long).fit()
print(model.summary()) 


# In[21]:


#same values for t-test and corresponding p-values for the effect of the type of iq than the previous t-test
stats.ttest_ind(data['FSIQ'], data['PIQ'])   


# ### 3.1.3.2. Multiple Regression: including multiple factors

# In[22]:


data = pandas.read_csv('iris.csv')
model = ols('sepal_width ~ name + petal_length', data).fit()
print(model.summary()) 


# ### 3.1.3.3. Post-hoc hypothesis testing: analysis of variance (ANOVA)

# In[23]:


#testing the difference between the coefficient associated to versicolor and virginica in the linear model  ANOVA
print(model.f_test([0, 1, -1, 0]))  
#NOT a significant difference


# ### 3.1.4. More visualization: seaborn for statistical exploration

# In[24]:


#using seaborn to produce a scattermatrix on continous variables
#firstly importing seaborn
import os
import urllib.request
if not os.path.exists('wages.txt'):
    urllib.request.urlretrieve('http://lib.stat.cmu.edu/datasets/CPS_85_Wages', 'wages.txt')

names = [
    'EDUCATION: Number of years of education',
    'SOUTH: 1=Person lives in South, 0=Person lives elsewhere',
    'SEX: 1=Female, 0=Male',
    'EXPERIENCE: Number of years of work experience',
    'UNION: 1=Union member, 0=Not union member',
    'WAGE: Wage (dollars per hour)',
    'AGE: years',
    'RACE: 1=Other, 2=Hispanic, 3=White',
    'OCCUPATION: 1=Management, 2=Sales, 3=Clerical, 4=Service, 5=Professional, 6=Other',
    'SECTOR: 0=Other, 1=Manufacturing, 2=Construction',
    'MARR: 0=Unmarried,  1=Married'
]
short_names = [n.split(':')[0] for n in names]
data = pandas.read_csv('wages.txt', skiprows=27, skipfooter=6, sep=None, header=None, engine='python')
data.columns = short_names
print(data)

import seaborn
seaborn.pairplot(data, vars=['WAGE', 'AGE', 'EDUCATION'],
                 kind='reg')
seaborn.pairplot(data, vars=['WAGE', 'AGE', 'EDUCATION'],
                 kind='reg', hue='SEX')  


# In[25]:


#resetting seaborn default
from matplotlib import pyplot as plt
plt.rcdefaults()


# In[26]:


#regression capturing relationship between wage and eduction
seaborn.lmplot(y='WAGE', x='EDUCATION', data=data) 


# ### 3.1.5. Testing for interactions

# In[27]:


#do wages increase more with educatoin in femlaes vs males
print(data)
import statsmodels.api as sm
formula = 'WAGE ~ EDUCATION + SEX + EDUCATION * SEX'
result = sm.OLS.from_formula(formula, data=data).fit()
print(result.summary()) 

