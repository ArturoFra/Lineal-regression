#!/usr/bin/env python
# coding: utf-8

# # Tratamiento de variables categóricas
# 

# In[2]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# In[ ]:


df = pd.read_csv("C:/Users/A Emiliano Fragoso/Desktop/MLcourse/python-ml-course-master/datasets/ecom-expense/Ecom Expense.csv")


# In[ ]:





# In[ ]:


df.head()


# In[ ]:





# In[ ]:


dummy_gender = pd.get_dummies(df["Gender"], prefix = "Gender")
dummy_city_tier = pd.get_dummies(df["City Tier"], prefix = "CityT")


# In[ ]:





# In[ ]:


dummy_gender, dummy_city_tier


# In[ ]:





# In[ ]:


column_names=df.columns.values.tolist()
column_names


# In[ ]:





# In[ ]:


df_new = df[column_names].join(dummy_gender)

df_new.head()


# In[ ]:





# In[ ]:


column_names = df_new.columns.values.tolist()


# In[ ]:





# In[ ]:


df_new = df_new[column_names].join(dummy_city_tier)


# In[ ]:





# In[ ]:


df_new.head()


# In[ ]:





# In[ ]:


feature_cols = ["Monthly Income", "Transaction Time", 
                "Gender_Female", "Gender_Male", "CityT_Tier 1", "CityT_Tier 2", "CityT_Tier 3", "Record"]


# In[ ]:





# In[ ]:


X = df_new[feature_cols]
Y= df_new["Total Spend"]


# In[ ]:





# In[ ]:


lm = LinearRegression()


# In[ ]:


lm.fit(X,Y)


# In[ ]:





# In[ ]:


lm.intercept_


# In[ ]:


lm.coef_


# In[ ]:


list(zip(feature_cols,lm.coef_))


# In[ ]:


lm.score(X,Y)


# El modelo puede ser escrito as: Total_spend = -79.41713030137453 + 'Monthly Income'* 0.14753898049205744 + 'Transaction Time' * 0.154946125495895 + 'Gender_Female' * -131.02501325554596 + 'Gender_Male'* 131.02501325554596 + 'CityT_Tier 1' * 76.76432601049525 + 'CityT_Tier 2'* 55.13897430923255 + 'CityT_Tier 3'* -131.90330031972792 + 'Record'* 772.1492053631357

# In[ ]:


df_new["Prediction"] = -79.41713030137453 + df_new['Monthly Income']* 0.14753898049205744 + df_new['Transaction Time'] * 0.154946125495895 + df_new['Gender_Female'] * -131.02501325554596 + df_new['Gender_Male'] * 131.02501325554596 + df_new['CityT_Tier 1'] * 76.76432601049525 + df_new['CityT_Tier 2'] * 55.13897430923255 + df_new['CityT_Tier 3'] * -131.90330031972792 + df_new['Record'] * 772.1492053631357 


# In[ ]:


df_new["Prediction2"] = lm.predict(pd.DataFrame(df_new[feature_cols]))


# In[ ]:





# In[ ]:





# In[ ]:


df_new.head()


# In[ ]:





# In[ ]:





# In[ ]:


SSD = sum((df_new["Prediction"] - df_new["Total Spend"])**2)


# In[ ]:





# In[ ]:





# In[ ]:


SSD


# In[ ]:





# In[ ]:





# In[ ]:


RSE = np.sqrt(SSD/(len(df_new)-len(feature_cols)-1))


# In[ ]:





# In[ ]:


RSE


# In[ ]:





# In[ ]:


sale_mean = np.mean(df_new["Total Spend"])


# In[ ]:





# In[ ]:


sale_mean


# In[ ]:





# In[ ]:


error = RSE/sale_mean


# In[ ]:





# In[ ]:


error


# In[ ]:





# # Eliminar variables dummies 

# In[ ]:


dummy_gender = pd.get_dummies(df["Gender"], prefix = "Gender").iloc[:,1:]
dummy_gender


# In[ ]:


dummy_city_tier = pd.get_dummies(df["City Tier"], prefix = "City").iloc[:,1:]
dummy_city_tier


# In[ ]:


column_names = df.columns.values.tolist()
df_new  = df[column_names].join(dummy_gender)
column_names = df_new.columns.values.tolist()
df_new  = df_new[column_names].join(dummy_city_tier)
df_new.head()


# In[ ]:


feature_cols = ["Monthly Income", "Transaction Time", "Gender_Male", "City_Tier 2", "City_Tier 3", "Record"]
X = df_new[feature_cols]
Y = df_new["Total Spend"]
lm = LinearRegression()
lm.fit(X,Y) 


# In[ ]:


lm.intercept_


# In[ ]:


list(zip(feature_cols, lm.coef_))


# In[ ]:


lm.score(X,Y)


# # Transformación de variables para conseguir una relación no lineal

# In[3]:


import pandas as pd


# In[4]:


data_auto = pd.read_csv("C:/Users/A Emiliano Fragoso/Desktop/MLcourse/python-ml-course-master/datasets/auto/auto-mpg.csv")


# In[5]:


data_auto.head()


# In[6]:


import matplotlib.pyplot as plt


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
data_auto["mpg"]=data_auto["mpg"].dropna()
data_auto["horsepower"]=data_auto["horsepower"].dropna()
plt.plot(data_auto["horsepower"], data_auto["mpg"], "ro")
plt.xlabel("Caballos de potencia")
plt.ylabel("Consumo (millas por galeón)")
plt.title("HP vs Consumo")


# In[ ]:





# # Modelo de regresión lineal
# *mpg=a+b x hp 

# In[59]:


X=data_auto["horsepower"].fillna(data_auto["horsepower"].mean())
Y=data_auto["mpg"].fillna(data_auto["mpg"].mean())
X_data = X[:, np.newaxis]


# In[60]:


lm = LinearRegression()
lm.fit(X_data,Y)


# In[57]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(X,Y, "ro")


# In[61]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(X,Y, "ro")
plt.xlabel("Horsepower")
plt.ylabel("Millas per Galeón")
plt.plot(X, lm.predict(X_data), color = "blue")


# In[62]:


lm.score(X_data, Y)


# In[73]:


SSD = np.sum((Y-lm.predict(X_data))**2)


# In[74]:


SSD


# In[ ]:





# In[75]:


SSD = np.sum((Y-lm.predict(X_data))**2)
RSE = np.sqrt(SSD/405)
y_mean = np.mean(Y)
error = RSE/y_mean
SSD, RSE, error


# In[ ]:





# # Modelo de regresión cuadrática
# *mpg=a+b x hp^2 

# In[76]:


X_data = X**2


# In[77]:


X_data = X_data[:, np.newaxis]


# In[78]:


lm = LinearRegression()
lm.fit(X_data,Y)


# In[81]:


lm.score(X_data, Y)


# In[82]:


SSD = np.sum((Y-lm.predict(X_data))**2)
RSE = np.sqrt(SSD/405)
y_mean = np.mean(Y)
error = RSE/y_mean
SSD, RSE, error


# In[ ]:





# In[80]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(X,Y, "ro")
plt.xlabel("Horsepower")
plt.ylabel("Millas per Galeón")
plt.plot(X, lm.predict(X_data), color = "blue")


# In[ ]:





# # Modelo de regresión lineal y cuadrática
# *mpg=a+b x hp + c x hp^2

# In[84]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model


# In[85]:


poly = PolynomialFeatures(degree = 2)


# In[86]:


X_data = poly.fit_transform(X[:, np.newaxis])


# In[88]:


lm = linear_model.LinearRegression()
lm.fit(X_data, Y)


# In[89]:


lm.score(X_data, Y)


# In[ ]:





# # Problema de los Outliers

# In[91]:


plt.plot(data_auto["displacement"], data_auto["mpg"], "ro")


# In[ ]:





# In[118]:



X = data_auto["displacement"].fillna(data_auto["displacement"].mean())
X=X[:, np.newaxis]
Y = data_auto["mpg"].fillna(data_auto["displacement"].mean())
lm = LinearRegression()
lm.fit(X,Y)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[119]:


lm.score(X,Y)


# In[ ]:





# In[103]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(X,Y, "ro")
plt.xlabel("Displacement")
plt.ylabel("Millas per Galeón")
plt.plot(X, lm.predict(X), color = "blue")


# In[105]:


data_auto[(data_auto["displacement"]>250) & (data_auto["mpg"]>35)]


# In[107]:


data_auto[(data_auto["displacement"]>300) & (data_auto["mpg"]>25)]


# In[111]:


data_auto_clean = data_auto.drop([395,372,258,305])


# In[121]:


X = data_auto_clean["displacement"].fillna(data_auto_clean["displacement"].mean())
X=X[:, np.newaxis]
Y = data_auto_clean["mpg"].fillna(data_auto_clean["displacement"].mean())
lm = LinearRegression()
lm.fit(X,Y)


# In[ ]:





# In[122]:


lm.score(X,Y)


# In[ ]:





# In[123]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(X,Y, "ro")
plt.xlabel("Displacement")
plt.ylabel("Millas per Galeón")
plt.plot(X, lm.predict(X), color = "blue")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




