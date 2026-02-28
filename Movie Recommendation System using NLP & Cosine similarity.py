#!/usr/bin/env python
# coding: utf-8

# # Movie Recommendation System using NLP & Cosine Similarity
# 
# This project builds a content-based movie recommendation system using NLP techniques and cosine similarity.
# 
# The system recommends top 5 similar movies based on:
# - Genres
# - Keywords
# - Cast
# - Crew
# - Overview

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.porter import PorterStemmer
import pickle


# ## Data Loading
# 
# Dataset used: TMDB 5000 Movies Dataset

# In[3]:


credits=pd.read_csv('tmdb_5000_credits.csv')
movies=pd.read_csv('tmdb_5000_movies.csv')


# In[4]:


credits.head()


# In[5]:


movies.head()


# In[6]:


mov=movies.merge(credits, on='title')


# In[7]:


mov.head()


# ## Data Preprocessing
# 
# Will select relevant features and handle missing values.

# In[8]:


movies.shape


# In[9]:


credits.shape


# In[10]:


mov.shape


# In[11]:


mov.info()


# In[12]:


mov=mov[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[14]:


mov.head()


# In[16]:


mov.info()


# In[17]:


mov.duplicated().sum()


# ## Feature Engineering
# 
# Extracting names from JSON columns and combining into a single 'tags' column.

# In[18]:


mov.iloc[0].genres


# In[27]:


import ast
ast.literal_eval


# In[28]:


def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[29]:


mov['genres']=mov['genres'].apply(convert)


# In[30]:


mov['genres']


# In[31]:


mov.iloc[0].keywords


# In[33]:


mov['keywords']=mov['keywords'].apply(convert)


# In[34]:


mov.head()


# In[35]:


mov.iloc[0].cast


# In[36]:


# Considering only top 3 names of crew & Director from cast


# In[37]:


def convert3(obj):
    L=[]
    c=0
    for i in ast.literal_eval(obj):
        if c!=3:
            L.append(i['name'])
            c=c+1
    return L


# In[39]:


mov['cast']=mov['cast'].apply(convert3)


# In[40]:


mov['cast']


# In[41]:


mov.iloc[0].crew


# In[42]:


def fetch_dirc(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break
    return L


# In[43]:


mov['crew']=mov['crew'].apply(fetch_dirc)


# In[44]:


mov.head()


# In[45]:


mov['overview'][0]


# In[47]:


mov.head()


# In[49]:


mov['overview'] = mov['overview'].fillna('')
mov['overview'] = mov['overview'].apply(lambda x: x.split())


# In[50]:


mov.head()


# In[51]:


# replacing gap between words for ex- Science fiction --- ScienceFiction so that model do not consider tehm as different


# In[52]:


mov['genres'].apply(lambda x: [i.replace(" ","") for i in x])


# In[53]:


mov['genres']=mov['genres'].apply(lambda x: [i.replace(" ","") for i in x])
mov['keywords']=mov['keywords'].apply(lambda x: [i.replace(" ","") for i in x])
mov['cast']=mov['cast'].apply(lambda x: [i.replace(" ","") for i in x])
mov['crew']=mov['crew'].apply(lambda x: [i.replace(" ","") for i in x])


# In[54]:


mov.head()


# In[55]:


# Adding all the contents into 1 text column


# In[64]:


mov['tags1']=mov['overview']+mov['genres']+ mov['keywords']+mov['cast']+mov['crew']


# In[67]:


mov_rev=mov[['movie_id','title','tags1']]


# In[69]:


mov_rev


# In[71]:


mov_rev['tags1']=mov_rev['tags1'].apply(lambda x:" ".join(x))


# In[72]:


mov_rev


# In[74]:


mov_rev['tags1'][0]


# In[75]:


mov_rev['tags1'][1]


# ## Text Processing (Stemming)
# 
# Converting words into their base form

# In[76]:


import sys
get_ipython().system('{sys.executable} -m pip install nltk')


# In[77]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[78]:


ps.stem('dancing')


# In[79]:


def stemmm(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)
        


# In[80]:


stemmm('In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. Action Adventure Fantasy ScienceFiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d JamesCameron')


# In[82]:


mov_rev['tags1']=mov_rev['tags1'].apply(stemmm)


# In[83]:


mov_rev


# ## Vectorization using Bag of Words

# In[84]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')


# In[86]:


vectors=cv.fit_transform(mov_rev['tags1']).toarray()


# In[87]:


cv.get_feature_names_out()


# In[88]:


vectors.shape


# ## Cosine Similarity 

# In[89]:


from sklearn.metrics.pairwise import cosine_similarity


# In[90]:


similarity=cosine_similarity(vectors)


# In[91]:


similarity[1]  # Shows similarity of movie 1 with other 4808 movies


# In[92]:


similarity.shape


# In[93]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# ## Recommendation Function to list similar 5 movies

# In[94]:


def recommend(movie):
    mov_index=mov_rev[mov_rev['title']==movie].index[0]
    distances= similarity[mov_index]
    movie_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]

    for i in movie_list:
        print(mov_rev.iloc[i[0]].title) 


# In[95]:


recommend('Avatar')


# ## Save Model Files

# In[96]:


pickle.dump(mov_rev.to_dict(),open('movies_dict.pkl','wb'))


# In[97]:


top_5_similarities = {}

for i in range(len(similarity)):

    sim_scores = list(enumerate(similarity[i]))

    # remove self similarity
    sim_scores = [x for x in sim_scores if x[0] != i]

    sim_scores = sorted(sim_scores,
                        key=lambda x: x[1],
                        reverse=True)[:5]

    top_5_similarities[i] = [j[0] for j in sim_scores]

import pickle
pickle.dump(top_5_similarities, open('similarity.pkl','wb'))


# In[ ]:




