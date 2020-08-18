#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nest_asyncio


# In[2]:


nest_asyncio.apply()
print("event loop is ready")


# In[3]:


config="config.yml"
training_files="data/"
domain="domain.yml"
output="models/"
print(config,training_files,domain,output)


# In[4]:


import rasa
model_path=rasa.train(domain,config,[training_files],output)
print(model_path)


# In[5]:


from rasa.jupyter import chat
chat(model_path)


# In[6]:


import rasa.data as data
stories_directory, nlu_data_directory=data.get_core_nlu_directories(training_files)
print(stories_directory, nlu_data_directory)


# In[7]:


rasa.test(model_path, stories_directory, nlu_data_directory)
print("done testing")


# In[ ]:




