
# coding: utf-8

# In[1]:


"""
    Steps for data cleaning and Exploratory Data Analysis
    1. Check if training record is unique.
    2. Check if Training and Test datasets are distinct.
    3. Check if data contains null values.
    4. Check the data type information
"""

import pandas as pd
import numpy as np
import csv as csv


# In[2]:


train_dataset = pd.read_csv('data/titanic/train.csv')
test_dataset = pd.read_csv('data/titanic/test.csv')


# In[3]:


train_dataset


# In[4]:


test_dataset.shape


# In[5]:


# check if the passengerId is unique
print("ID is unique") if train_dataset.PassengerId.nunique() == train_dataset.shape[0] else print("Invalid data")

# check if the training dataset and testing dataset are distinct
intersection = np.intersect1d(train_dataset.PassengerId.values, test_dataset.PassengerId.values)
print("Training and test data are distinct") if len(intersection) == 0 else print("wrong")

data_has_nan = False
if train_dataset.count().min() == train_dataset.shape[0] and test_dataset.count().min() == test_dataset.shape[0]:
    print("we need not to worry about nan")
else:
    data_has_nan = True
    print("data has nan")


# In[6]:


print("Training dataset column type information")
dtype_df = train_dataset.dtypes.reset_index()
dtype_df.columns = ["count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()


# In[7]:


print("Training dataset info...")


# In[8]:


dtype_df


# In[9]:


if data_has_nan:
    nas = pd.concat([train_dataset.isnull().sum(), test_dataset.isnull().sum()], axis=1, keys=['Train Dataset', 'Test Dataset'], sort=True)
    print("Nan in the datasets")
    print(nas[nas.sum(axis=1) > 0])


# In[10]:


survived = train_dataset[train_dataset['Survived'] == 1]['Sex'].value_counts()
dead = train_dataset[train_dataset['Survived'] == 0]['Sex'].value_counts()
sf = pd.DataFrame([survived, dead])
sf.index = ['Survived', 'Dead']


# # import python lib for visualization 

# In[11]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()


# To Visualize categorical features we need to use bar chart. The data has following categorical features
# 1. Pclass
# 2. Sex
# 3. SibSp
# 4. Parch
# 5. Embarked
# 6. Cabin

# In[12]:


def bar_chart(feature):
    survived = train_dataset[train_dataset['Survived'] == 1][feature].value_counts()
    dead = train_dataset[train_dataset['Survived'] == 0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['Survived', 'Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))


# In[13]:


bar_chart('Sex')


# # This chart confirms that female has more chance of getting survived

# In[14]:


bar_chart('Pclass')


# This chart confirms that class 1 has more chance of survival and class 3 has more chance of being.

# In[15]:


bar_chart('SibSp')


# In[16]:


bar_chart('Parch')


# In[17]:


bar_chart('Embarked')


# # Feature engineering
# Feature engineering is the process of using domain knowledge of the data
# to create features (feature vectors) that make machine learning algorithms work.
# 
# feature vector is an n-dimensional vector of numerical features that represent some object.
# Many algorithms in machine learning require a numerical representation of objects,
# since such representations facilitate processing and statistical analysis 

# In[18]:


train_test_dataset = [train_dataset, test_dataset]
for data in train_test_dataset:
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)


# In[19]:


train_dataset['Title'].value_counts()


# In[20]:


title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for data in train_test_dataset:
    data['Title'] = data['Title'].map(title_mapping)


# In[21]:


train_dataset['Title'].value_counts()


# In[22]:


train_dataset.head()


# In[23]:


bar_chart('Title')


# In[24]:


train_dataset.drop('Name', axis=1, inplace=True)
test_dataset.drop('Name', axis=1, inplace=True)


# In[25]:


train_dataset


# In[26]:


# sex
sex_mapping = {'male': 0, 'female':1}
for dataset in train_test_dataset:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)


# In[27]:


train_dataset.head()


# In[28]:


bar_chart('Sex')


# In[29]:


train_dataset['Age'].fillna(train_dataset.groupby('Title')['Age'].transform('median'), inplace=True)
test_dataset['Age'].fillna(test_dataset.groupby('Title')['Age'].transform('median'), inplace=True)


# In[30]:


train_dataset.head()


# In[31]:


def fact_grid(feature, x_lim1=None, x_lim2=None):
    facet = sns.FacetGrid(train_dataset, hue="Survived",aspect=4)
    facet.map(sns.kdeplot,feature,shade= True)
    facet.set(xlim=(0, train_dataset[feature].max()))
    facet.add_legend()
    if x_lim1 != None and x_lim2 != None:
        plt.xlim(x_lim1, x_lim2)


# In[32]:


fact_grid('Age')


# In[33]:


fact_grid('Age', 0, 20)


# In[34]:


fact_grid('Age', 20, 30)


# In[35]:


fact_grid('Age', 30, 40)


# In[36]:


fact_grid('Age', 40, 60)


# In[37]:


fact_grid('Age', 60, 100)


# # Binning -> Converting numerical age to categorical variable

# In[38]:


for dataset in train_test_dataset:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4


# In[39]:


bar_chart('Age')


# ## Embarked

# In[40]:


p_class1 = train_dataset[train_dataset['Pclass'] == 1]['Embarked'].value_counts()
p_class2 = train_dataset[train_dataset['Pclass'] == 2]['Embarked'].value_counts()
p_class3 = train_dataset[train_dataset['Pclass'] == 3]['Embarked'].value_counts()
df = pd.DataFrame([p_class1, p_class2, p_class3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))


# ## since port 'S' is in majority in all 3 cases, so we can fill null value by 'S'

# In[41]:


for dataset in train_test_dataset:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    
embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test_dataset:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)


# In[42]:


train_dataset.head()


# In[43]:


train_dataset['Fare'].fillna(train_dataset.groupby('Pclass')['Fare'].transform("median"), inplace=True)
test_dataset['Fare'].fillna(test_dataset.groupby('Pclass')['Fare'].transform("median"), inplace=True)


# In[44]:


train_dataset.head(50)


# In[45]:


fact_grid('Fare', 17, 30)


# In[46]:


for dataset in train_test_dataset:
    dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,
    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3


# In[47]:


train_dataset.head()


# In[48]:


for dataset in train_test_dataset:
    dataset['Cabin'] = dataset['Cabin'].str[:1]
p_class1_cabin = train_dataset[train_dataset["Pclass"] == 1]['Cabin'].value_counts()
p_class2_cabin = train_dataset[train_dataset["Pclass"] == 2]['Cabin'].value_counts()
p_class3_cabin = train_dataset[train_dataset["Pclass"] == 3]['Cabin'].value_counts()
df = pd.DataFrame([p_class1_cabin, p_class2_cabin, p_class3_cabin])
df.index = ['1st class','2nd class', '3rd class']
#df.plot(kind='bar',stacked=True, figsize=(20,10))
p_class1_cabin


# In[49]:


p_class2_cabin


# In[50]:


p_class3_cabin


# In[51]:


176 + 12 + 16


# In[52]:


204/8


# In[53]:


15/25.5


# In[54]:


32/25.5


# In[55]:


cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in train_test_dataset:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)


# In[56]:


train_dataset["Cabin"].fillna(train_dataset.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test_dataset["Cabin"].fillna(test_dataset.groupby("Pclass")["Cabin"].transform("median"), inplace=True)


# In[57]:


train_dataset["FamilySize"] = train_dataset["SibSp"] + train_dataset["Parch"] + 1
test_dataset["FamilySize"] = test_dataset["SibSp"] + test_dataset["Parch"] + 1


# In[58]:


fact_grid('FamilySize')


# In[59]:


family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
for dataset in train_test_dataset:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)


# In[60]:


train_dataset.head()


# In[61]:


features_drop = ['Ticket', 'SibSp', 'Parch']
train_dataset = train_dataset.drop(features_drop, axis=1)
test_dataset = test_dataset.drop(features_drop, axis=1)
train_dataset = train_dataset.drop(['PassengerId'], axis=1)


# In[62]:


train_data = train_dataset.drop('Survived', axis=1)
target = train_dataset['Survived']

train_data.shape, target.shape


# In[63]:


train_data.head()


# In[64]:


train_dataset.head()


# In[65]:


from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# In[66]:


clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[67]:


round(np.mean(score)*100,2)


# In[69]:


clf = SVC()
clf.fit(train_data, target)

test_data = test_dataset.drop("PassengerId", axis=1).copy()
prediction = clf.predict(test_data)


# In[71]:


submission = pd.DataFrame({
        "PassengerId": test_dataset["PassengerId"],
        "Survived": prediction
    })

submission.to_csv('submission.csv', index=False)


# In[72]:


submission = pd.read_csv('submission.csv')
submission.head()

