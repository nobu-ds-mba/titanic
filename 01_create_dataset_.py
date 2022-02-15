#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd

train_data = pd.read_csv('/Users/nobu/Jupyter_lab/Kaggle/Titanic/data/input/train.csv')
train_data.head()


# Name：敬称あり  
# Ticket，Cabin，Embarked：記号あり

# In[10]:


test_data = pd.read_csv('/Users/nobu/Jupyter_lab/Kaggle/Titanic/data/input/test.csv')
test_data.head()


# In[11]:


train_data.describe()


# In[17]:


# trainとtestを縦に結合し，全データで確認
full_data = pd.concat([train_data, test_data], axis = 0, sort = False)
print(full_data.shape)


# In[18]:


full_data.describe()


# 生存割合：4割弱  
# 年齢の中央値：28歳，高齢者少なそう

# In[19]:


# オブジェクト型の要素数describeの引数include = 'O'
full_data.describe(include = 'O')


# 名前は1309個中，1307個がユニーク＝ほぼ全員異なる名前  
# チケットは数百の重複あり

# In[ ]:


pip install pandas_profiling


# In[25]:


# 各特徴料を個別に把握：pandas-profiling
import pandas_profiling as pdp
display(php.__version__)
pdp.ProfileReport(train_data)


# In[12]:


# 欠損度合い確認
train_data.isnull().sum()


# 年齢情報，cabin情報欠損多い  
# 年齢は補完可能かも

# In[13]:


test_data.isnull().sum()


# 年齢情報は要補完

# In[15]:


# 相関関係をプロット
import seaborn as sns
sns.pairplot(train_data)


# In[1]:


# 各特徴とtargetとの関係を可視化
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().system('pip install japanize-matplotlib')
import japanize_matplotlib


# In[13]:


# trainの死亡者と生存者別に集計
sns.countplot(x = 'Survived', data = train_data)
plt.title('死亡者，生存者の数')
plt.xticks([0, 1], ['死亡者', '生存者'])
plt.show()


# In[14]:


display(train_data['Survived'].value_counts())


# In[15]:


display(train_data['Survived'].value_counts() / len(train_data['Survived']))


# In[ ]:




