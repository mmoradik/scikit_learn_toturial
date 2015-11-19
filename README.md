# scikit_learn_toturial
from sklearn import datasets
#scikit-learn comes with a few standard datasets, 
#for instance the iris and digits datasets for classification and 
#the boston house prices dataset for regression.
iris=datasets.load_iris()
digits=datasets.load_digits()
print(digits.data)
print(shape(digits.data))
shape(digits.target)


# In[20]:

digits.target
print(digits.target)


# In[7]:

digits.images[0]


# In[11]:

from sklearn import svm
clf=svm.SVC(gamma=0.001, C=100.)
clf.fit(digits.data[:-1],digits.target[:-1])


# In[23]:


clf.predict(digits.data[-1:])
