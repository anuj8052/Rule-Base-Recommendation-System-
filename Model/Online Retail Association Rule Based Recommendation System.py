#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

# #### Imported important libraries for the data preprocessing as well as model building

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns


# 1. the mlxtend library that provides implementations of various frequent pattern mining algorithms, including the Apriori algorithm.

# In[2]:


pip install mlxtend


# In[3]:


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules


# ### Data Prerocessing

# ##### accessing the dataset from the online retail datasheet for the year 2010-2011

# In[4]:


df = pd.read_excel("Downloads/online_retail_II.xlsx", sheet_name = 'Year 2010-2011')


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df.shape


# In[8]:


df.describe()


# In[9]:


# !pip install openpyxl


# In[10]:


# checking for the unique values in each column for the futher analysis


# In[11]:


df['Quantity'].nunique()


# In[12]:


df['StockCode'].unique()


# In[13]:


df['StockCode'].nunique()


# In[14]:


df['Country'].unique()


# In[15]:


df['Country'].nunique()


# In[16]:


df['Customer ID'].unique() 


# In[17]:


df['Customer ID'].nunique()


# ### Checking for Missing or NaN values

# In[25]:


df.isna().sum()


# 1. As we can see in the Description Column there 1454 NaN values
# 2. for the column Customer ID there are 135080 NaN vlaues or we can say that 135080 rows
# 3. To handle the missing values or NaN values we can directly drop or fill while seeing previous data

# In[26]:


df.isnull().head()


# ### creating a mask i.e boolian mask for those rows which has NaN or null values
# 1. extracting alll the rows having NaN or missing values

# In[27]:


mask = df.isnull().any(axis = 1)
rows_with_nan = df[mask]


# In[28]:


rows_with_nan


# In[29]:


rows_with_nan.shape


# 1. As we can see that we have NaN value rows more than 1 lakh i.e 135080
# 2. For our analysis we will check for the upper outlier and lower outlier vlaues in the dataset and according to we will drop the values

# In[30]:


rows_with_nan.head()


# In[31]:


rows_with_nan.info()


# In[32]:


data = df.copy()


# In[33]:


data.head()


# In[34]:


data.shape


# In[35]:


data.dropna(inplace = True)


# In[36]:


data.isna().sum()


# In[37]:


data.shape


# ### Data Preprocessing and Cleaning Part

# In[38]:


def outlier_threshold(dataframe, variable):
    q1 = dataframe[variable].quantile(0.01)
    q3 = dataframe[variable].quantile(0.99)
    iqr = q3 - q1
    multiplier = 1.5

    upper_limit = q3 + (multiplier * iqr)
    lower_limit = q1 - (multiplier * iqr)

    return lower_limit, upper_limit


# In the above code we calculates the lower and upper limits for outliers based on the 1st percentile (q1) and 99th percentile (q3) of the variable's values. The interquartile range (iqr) is computed as the difference between q3 and q1. The multiplier factor of 1.5 is then applied to the iqr to determine the threshold range. Finally, the lower and upper limits are returned as a tuple.

# In[39]:


def replace_with_threshold(dataframe, variable):
    low_limit, up_limit = outlier_threshold(dataframe, variable)
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit


# In the above code,  the function first obtains the lower and upper limits using the outlier_threshold function. Then, it replaces the values in the variable column that are below the low_limit with the low_limit value using the loc accessor. Similarly, it replaces the values above the up_limit with the up_limit value. The modifications are made directly in the DataFrame.

# In[40]:


def retail_data_prep(dataframe):
    dataframe = dataframe[~(dataframe['StockCode'] == 'POST')]
    dataframe = dataframe.dropna()
    dataframe = dataframe[~(dataframe['Invoice'].str.contains('C', na=False))]
    dataframe = dataframe[dataframe['Price'] > 0]

    replace_with_threshold(dataframe, 'Price')
    replace_with_threshold(dataframe, 'Quantity')

    return dataframe


# In the above code,  the function removes rows where the StockCode column contains the value 'POST'. It then drops rows with missing values using dropna(). Next, it removes rows where the 'Invoice' column contains the letter 'C', indicating canceled invoices. After that, it filters out rows where the 'Price' column is less than or equal to 0.
# 
# Finally, it applies the replace_with_threshold function to handle outliers for the 'Price' and 'Quantity' columns. The modified DataFrame is returned at the end of the function.

# In[41]:


df = retail_data_prep(df)


# "df", is the modified DataFrame for the further process

# In[42]:


df.head()


# In[43]:


# using info() function in pandas, we can find the cruecial information related to the dataset like summary of whole
# dataset exmpl: columns name, their counts, their datatypes etc.
df.info()


# In[44]:


df.shape  #it tells the number of rows and columns in the dataset


# In[45]:


df.isna().sum() #this function used to find if there are missing values of NaN values in the dataset


# In[46]:


df.columns


# In[47]:


# df1 for ploting the plot to see the relationship between the features of the data


# In[48]:


df1 = df[['Quantity', 'InvoiceDate', 'Price', 'Customer ID']]


# In[49]:


df1.head()


# In[50]:


# df1.plot()
import seaborn as sns


# After data cleaning process we are seeing the unique values for further analysis

# In[51]:


df['Customer ID'].nunique()


# In[52]:


df['Description'].unique()


# In[53]:


df['Description'].nunique()


# In[54]:


df['Quantity'].nunique()


# In[55]:


df.describe()


# ### Data Visualization Process to get some insight
# 
# 1. Histogram
# * A histogram can help you understand the distribution of numerical variables like Quantity or Price. This will allow you to observe patterns in the data and identify any outliers or unusual behavior.

# In[56]:


#the distribution of numerical variables like Quantity or Price
variable = 'Price'

# Created the histogram
plt.figure(figsize=(8, 5))
plt.hist(df[variable], bins=30, edgecolor='black')

plt.title('Histogram of ' + variable)
plt.xlabel(variable)
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[57]:


variable = 'Quantity'

# Created the histogram
plt.figure(figsize=(8, 5))
plt.hist(df[variable], bins=30, edgecolor='black')

plt.title('Histogram of ' + variable)
plt.xlabel(variable)
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# 2. Scatter Plot between Quantity vs Price to see the behavour of intms over price

# In[58]:


# scatter plot to see the replationship between quantity and price of the items
x = 'Quantity'  
y = 'Price'  

# Creating the scatter plot
plt.figure(figsize=(8, 4))
plt.scatter(df[x], df[y])

plt.title('Scatter Plot: ' + x + ' vs ' + y)
plt.xlabel(x)
plt.ylabel(y)
plt.grid(True)

# to display the plot
plt.show()


# In[59]:


# df.info()
df.head()


# In[60]:


# time series visualization on Price of items during time like monthly, yearly, daily etc.
df.set_index('InvoiceDate', inplace=True)


# In[61]:


# Resampling the data by a specific time period, e.g., monthly or daily
resampled_data = df.resample('M').sum()
# Create the time series plot
plt.figure(figsize=(6, 3))
plt.plot(resampled_data.index, resampled_data['Price'])
# Customize the plot
plt.title('Monthly Sales Trend')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True)
plt.show()


# In[62]:


df.reset_index()


# ### Model Building

# ##### Preparing Association Rule Learning Data Structures and Association Rules Analysis
# ##### 1. Apriori Algorithm

# Apriori Algorithm is a technique used for market basket analysis, which aims to discover associations between items that are frequently purchased together.
# 
# Metrics:
# 1) Support(X, Y) = Freq(X, Y) / N
# 
# Support measures the probability of the co-occurrence of items X and Y in the transactions. It is calculated by dividing the frequency of transactions containing both X and Y by the total number of transactions.
# 
# 2) Confidence(X, Y) = Freq(X, Y) / Freq(X)
# 
# Confidence measures the conditional probability of item Y being purchased given that item X is already purchased. It is calculated by dividing the frequency of transactions containing both X and Y by the frequency of transactions containing X.
# 
# 3) Lift = Support(X, Y) / (Support(X) * Support(Y))
# 
# Lift quantifies the strength of the association between items X and Y. It compares the observed support of X and Y occurring together to what would be expected if X and Y were statistically independent. A lift value greater than 1 indicates a positive association, meaning the occurrence of item X increases the likelihood of item Y being purchased.
# 
# These metrics help in identifying meaningful associations between items in a dataset. By analyzing support, confidence, and lift values, one can uncover item associations that can be utilized for various purposes such as targeted marketing, product placement, or recommendation systems.
# 
# * Generally, association rules are written in “IF-THEN” format. We can also use the term “Antecedent” for IF (LHS) and “Consequent” for THEN (RHS)

# In[23]:


def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', 'StockCode'])['Quantity'].sum().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)

def create_rules(dataframe, id=True, country='Germany'):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='support', min_threshold=0.01)
    return rules


# In[24]:


recom_rules = create_rules(df)


# In[25]:


recom_rules.sort_values(by = 'lift', ascending = False).head(10)


# ### Output for User Recommendation

# In[26]:


# def check_id(df, stock_code):
#     i = 0
#     product_name = None
#     while i < len(df):
#         if df.loc[i, 'StockCode'] == stock_code:
#             product_name = df.loc[i, 'Description']
#             break
#         i += 1
#     return product_name

def check_id(df, stock_code):
    matching_rows = df[df['StockCode'] == stock_code]
    product_name = matching_rows['Description'].values[0] if not matching_rows.empty else None
    return product_name


# In[27]:


# def arl_recommender(rules_df, product_id):
#     sorted_rules = rules_df.sort_values(by='lift', ascending=False)
#     idx = 0
#     while idx < len(sorted_rules):
#         product = sorted_rules.iloc[idx]['antecedents']
#         for j in list(product):
#             if j == product_id:
#                 return list(sorted_rules.iloc[idx]['consequents'])[0]
#         idx += 1

def arl_recommender(rules_df, product_id):
  sorted_rules = rules_df.sort_values(by='lift', ascending=False)
  for idx, product in enumerate(sorted_rules['antecedents']):
    for j in list(product):
      if j == product_id:
        return list(sorted_rules.iloc[idx]['consequents'])[0]


# In[28]:


# product_list = [21987, 23235, 22747]
# recom_list = []

# for product in product_list:
#   recom_list.append(arl_recommender(recom_rules, product))
# product_list = [21987, 23235, 22747, 71053]
product_list = [21987]
recom_list = []

for product in product_list:
  recom_list.append(arl_recommender(recom_rules, product))


# In[29]:


df.head()


# In[30]:


df['StockCode'].unique()


# In[31]:


df[df['StockCode'] == 21987]


# In[32]:


df[df['StockCode'] == 71053].nunique()


# In[33]:


df[df['StockCode'] == 71053]


# In[34]:


df.info()


# In[35]:


recom_list


# In[36]:


for product in product_list:
  print(check_id(df, product))


# In[37]:


for recom in recom_list:
  print(check_id(df, recom))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




