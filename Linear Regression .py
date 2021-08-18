#!/usr/bin/env python
# coding: utf-8

# # Modelos de Regresión Lineal

# ## Modelo con datos simulados 
# y=a + bx
# x: 100 valores distribuidos según una N(1.5,2.5)
# Ye= 5 + 0.9 * x + e
# e estará distribuida según N(0 , 0.8)

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


x = 1.5 + 2.5*np.random.randn(100)


# In[4]:


res = 0 + 0.8*np.random.randn(100)


# In[5]:


y_pred = 5 + 0.9*x


# In[28]:


y_act = 5 + 0.9*x + res


# In[10]:


x_list=x.tolist()
y_pred_list = y_pred.tolist()
y_act_list = y_act.tolist()


# In[12]:


data = pd.DataFrame(
    {
        "x" : x_list,
        "y_actual" : y_act_list,
        "y_pred" : y_pred_list
        
        
    }



)


# In[ ]:





# In[13]:


data.head()


# In[19]:


import matplotlib.pyplot as plt


# In[21]:


y_mean = [np.mean(y_act) for i in range (1, len(x_list) + 1)]


# In[ ]:





# In[22]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(x,y_pred)
plt.plot(x,y_act, "ro")
plt.plot(x,y_mean)
plt.title("Valor Actual vs Predicción")


# In[30]:


data["SSR"]=(data["y_pred"] - np.mean(y_act))**2
data["SSD"]=(data["y_pred"] - data["y_actual"])**2
data["SST"]=(data["y_actual"] - np.mean(y_act))**2


# In[32]:


data.head()


# In[33]:


SSR= sum(data["SSR"])
SSD= sum(data["SSD"])
SST= sum(data["SST"])


# In[34]:


SSR


# In[35]:


SSD


# In[36]:


SST


# In[37]:


SSR+SSD


# In[38]:


r2=SSR/SST


# In[39]:


r2


# In[41]:


plt.hist((data["y_pred"] - data["y_actual"]))


# # Obteniendo la recta de la regresión

# In[43]:


x_mean = np.mean(data["x"])
y_mean = np.mean(data["y_actual"])
x_mean, y_mean


# In[45]:


data["beta_n"]= (data["x"]-x_mean)*(data["y_actual"]-y_mean)  ##Covariancia
data["beta_d"]= (data["x"]-x_mean)**2


# In[46]:


beta = (sum(data["beta_n"]))/(sum(data["beta_d"]))


# In[47]:


alpha = y_mean - beta*x_mean


# In[48]:


alpha, beta


# El modelo obtenido por regresión es : y = 5.04135919 + 0.8935919893 * x

# In[49]:


data["y_model"]= alpha + beta * data["x"]


# In[50]:


data.head()


# In[51]:


SSR=sum((data["y_model"] - np.mean(y_mean))**2)
SSD=sum((data["y_model"] - data["y_actual"])**2)
SST=sum((data["y_actual"] - np.mean(y_mean))**2)


# In[52]:


SSR,SSD,SST


# In[54]:


r2=SSR/SST
r2


# In[62]:


y_mean = [np.mean(y_act) for i in range (1, len(x_list) + 1)]
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(x,y_pred, "b")
plt.plot(x,y_act, "ro")
plt.plot(x,y_mean, "g")
plt.plot(data["x"],data["y_model"], "r")
plt.title("Valor Actual vs Predicción")


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




