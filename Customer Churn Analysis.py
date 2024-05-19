#!/usr/bin/env python
# coding: utf-8

# # Customer Churn Analysis
# This is a dataset of an ecommerce company and we have some customers who are churning (leaving).

# ## Goals
# 1. Perform exploratory analysis of the provided customer data to share insights of the behavior and characteristics of the customers. Make suggestions to help the company with customer retention
# 
# 2. Build a predictive model to identify customers who are at risk of leaving the company (churn) based on the provided variables. This can help the company take proactive steps to retain these customers and reduce the rate of churn

# ## Data description
# 1. CustomerID
# 
# 2. Churn: Churn Flag
# 
# 3. Tenure: in months
# 
# 4. PreferredLoginDevice
# 
# 5. CityTier
# 
# 6. WarehouseToHome: Distance in between warehouse to home of customer
# 
# 7. PreferredPaymentMode
# 
# 8. Gender
# 
# 9. HourSpendOnApp
# 
# 10. NumberOfDeviceRegistered
# 
# 11. PreferedOrderCat
# 
# 12. SatisfactionScore
# 
# 13. MaritalStatus
# 
# 14. NumberOfAddress
# 
# 15. OrderAmountHikeFromlastYear: Percentage increases in order from last year
# 
# 16. CouponUsed: Total number of coupon has been used in last month
# 
# 17. OrderCount: Total number of orders has been places in last month
# 
# 18. DaySinceLastOrder
# 
# 19. CashbackAmount: Average cashback in last month
# 
# 20. Complain: Complain flag - if the customer ever had a complain

# ## Tips
# 1. Sharing your thoughts and reasoning as you go will help!
# 2. Feel free to use any libraries like scikit-learn, use stackoverflow. Don't use ChatGPT and similar.
# 3. If you are unable to complete any step I can provide help towards the answer. Demonstrating understanding of the solution and result will earn some points!

# ## Import the libraries

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# # 1. Data overview

# #### Question: Read the sheet named 'E Comm' from file 'E Commerce Dataset.xlsx' saved in current directory into df variable. Print first 5 rows of the dataframe

# In[2]:


df = pd.read_excel('E Commerce Dataset.xlsx')
df.head(5)


# In[3]:


df.shape


# In[4]:


df.info()


# #### Question: How many unique values are in each column?

# In[5]:


for columns in df:
  print(columns,df[columns].nunique())


# #### Question: Calculate average churn rate
# Note: Churn = 1 means customer has churned

# In[6]:


df[df['Churn'] == 1].count()


# In[7]:


#check if there is any n/a
df['Churn'].isna().nunique()


# In[8]:


#churn rate
df[df['Churn'] == 1].shape[0]/df.shape[0]*100


# #### Question: How many missing values / nulls are there in each column?

# In[9]:


for col in df.columns:
  print(col, df[col].isna().sum())


# # 2. Exploratory Data Analysis

# ## Univariate analysis
# Here we will understand select variables

# ### 1. Numeric variables

# #### Question: Histogram
# Show histograms for all numeric columns. Describe the each variable's distribution briefly to a business stakeholder

# In[10]:


numerical_df = df[['CustomerID','Churn','Tenure','CityTier','WarehouseToHome','HourSpendOnApp','NumberOfDeviceRegistered','SatisfactionScore','NumberOfAddress','Complain', 'OrderAmountHikeFromlastYear',	'CouponUsed',	'OrderCount',	'DaySinceLastOrder',	'CashbackAmount']]
numerical_df.head(5)


# In[11]:


categorical_df = df[['PreferredLoginDevice','PreferredPaymentMode','Gender','PreferedOrderCat','MaritalStatus']]
categorical_df.head(5)


# In[12]:


sns.histplot(data=numerical_df, x="CityTier")


# In[13]:


sns.pairplot(numerical_df,hue="Churn")


# ### 2. Non numeric columns

# Cleaning (ignore)

# In[14]:


#As mobile phone and phone are both same so we have merged them
df.loc[df['PreferredLoginDevice'] == 'Phone', 'PreferredLoginDevice' ] = 'Mobile Phone'
df.loc[df['PreferedOrderCat'] == 'Mobile', 'PreferedOrderCat' ] = 'Mobile Phone'
#as cod is also cash on delievery
#as cc is also credit card so i merged them
df.loc[df['PreferredPaymentMode'] == 'COD', 'PreferredPaymentMode' ] = 'Cash on Delivery'   # uses loc function
df.loc[df['PreferredPaymentMode'] == 'CC', 'PreferredPaymentMode' ] = 'Credit Card'


# #### Question: Show the unique values of each variable and print the number of occurences of each value
# Describe these results briefly

# In[15]:


for col in df.columns:
    if df[col].dtype == object:
      print(df[col].unique(), df[col].value_counts())


# ## Analysing the Churn by select variables
# Provide business recommendation for each of the below

# #### Question: Relation between complains and churn

# In[16]:


df['Churn'].corr(df['Complain'])


# In[20]:


churn_rate_complain = df[df['Complain'] == 1]['Churn'].mean()
churn_rate_no_complain = df[df['Complain'] == 0]['Churn'].mean()


# In[21]:


print("Churn rate for Complain =", churn_rate_complain)
print("Churn rate for No Complain =",churn_rate_no_complain)


# ## Correlation matrix
# Visualize the correlation between all variables

# #### Question: Do we need to do any preprocessing on categorical variables before calculating correlation?

# In[22]:


#Onc hot coding for categorical dataset
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
new_df = enc.fit_transform(categorical_df)
df_ohc = pd.DataFrame(new_df.toarray(), columns=enc.get_feature_names_out(), dtype=int)


# In[23]:


df_ohc.head(5)


# In[24]:


df_final = pd.concat([numerical_df, df_ohc], axis=1)
df_final.head(5)


# In[25]:


df_final = df_final.dropna()
df_final.head(5)


# #### Question: Plot correlation matrix
# Discuss a few significant correlations

# In[26]:


#CustomerID column as it's not relevant for correlation
df_final = df_final.drop(columns=['CustomerID'])
df_final.head(5)


# In[28]:


corr_matrix = df_final.corr()
corr_matrix


# In[31]:


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(figsize=(10, 8), dpi=150)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.8)
plt.show()


# #### Optional: What is the correlation of each feature with target
# Sort the correlation in descending order

# In[35]:


corr_matrix.sort_values(by = 'Churn', ascending=False)


# # 3. Modelling

# ## Prepare data
# Fill nulls in each column

# In[36]:


from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=2)


# In[37]:


for col in df.columns:
    nullCount = df[col].isnull().sum()
    if nullCount > 0:
        df[col]=imputer.fit_transform(df[[col]])
        print("Filled", nullCount, "nulls in", col)


# #### Question: Make the data suitable for model training

# In[39]:


df_final.head(5)


# In[40]:


from sklearn.linear_model import LogisticRegression
x = df_final.drop(['Churn'],axis = 1)
x.head(5)
y = df_final['Churn']


# In[47]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[48]:


#check the shape of X_train and X_test
x_train.shape, x_test.shape


# In[49]:


clf = LogisticRegression(random_state=0).fit(x_train,y_train)


# In[51]:


clf.fit(x_train, y_train)


# In[52]:


#predict result
y_pred = clf.predict(x_test)


# ## Model training

# #### Question: Train one or more models and show their performance on training and test data

# Train and show train & test accuracy

# In[54]:


from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score, confusion_matrix
train_accuracy = accuracy_score(y_pred, y_test)*100
train_accuracy


# In[56]:


y_pred_train = clf.predict(x_train)
y_pred_train


# In[57]:


print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))


# # 4. Evaluation
# Feel free to use any libraries / stackoverflow

# #### Question: Show the precision, recall and f1 score for the model with best accuracy
# 

# In[59]:


print(classification_report(y_pred, y_test, digits=6))


# print(classification_report(y_train, y_pred_train, digits=6))

# ## Confusion matrix

# #### Question: Show the confusion matrix
# Describe the performance of model and steps you could take to improve it

# #### Bonus: Plot the confusion matrix

# In[61]:


print('Confusion matrix:\n', confusion_matrix(y_pred, y_test))


# In[62]:


print('Confusion matrix:\n', confusion_matrix(y_pred_train, y_train))


# # Discussion

# In[64]:


#check for overfitting and underfitting
print('Training set score: {:.4f}'.format(clf.score(x_train, y_train)))


# In[65]:


print('Test set score: {:.4f}'.format(clf.score(x_test, y_test)))


# The training-set accuracy score is 0.8917 while the test-set accuracy to be 0.8927. These two values are quite comparable. So, there is no question of overfitting.

# Next Step, we could perform hyperparameter optimization using gridsearch CV to improve the performance for this particular model

# In[ ]:




