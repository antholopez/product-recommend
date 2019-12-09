#!/usr/bin/env python
# coding: utf-8

# In[110]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
from random import randrange
ds = pd.read_csv("articulos.csv", encoding='latin1')


# In[111]:


tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0)
tfidf_matrix = tf.fit_transform(ds['Marca'])


# In[112]:


cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix) 
results = {}
for idx, row in ds.iterrows():
   similar_indices = cosine_similarities[idx].argsort()[:-100:-1] 
   similar_items = [(cosine_similarities[idx][i], ds['id_articulo'][i]) for i in similar_indices] 
   results[row['id_articulo']] = similar_items[1:]


# In[113]:


def item(id):  
  return ds.loc[ds['id_articulo'] == id]['articulo'].tolist()[0]

# Just reads the results out of the dictionary.def 
def recommend(item_id, num):
    recs = results[item_id][:num]
    data = np.array(recs)
    response = []
    for i, rec in data:
        response.append(item(rec))
    return response
        


# In[158]:


art_id = randrange(717)
print(art_id)


# In[159]:


print(recommend(item_id=art_id, num=3))


# In[ ]:





# In[ ]:




