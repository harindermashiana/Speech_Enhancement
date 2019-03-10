#!/usr/bin/env python
# coding: utf-8

# In[14]:


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D,Conv2DTranspose
import keras
from keras import optimizers


# In[20]:


model=Sequential()
model.add(Convolution2D(256, 5, 5,activation='relu',
            init='he_uniform',
            input_shape=(1024,87,1)))
model.add(Dropout(0.5))
model.add(Convolution2D(256, 5, 5,activation='relu',
            init='he_uniform',
            ))
model.add(Dropout(0.5))
model.add(Convolution2D(512, 5, 5,activation='relu',
            init='he_uniform',
            ))
model.add(Dropout(0.5))
model.add(Conv2DTranspose(512,5,5))
model.add(Conv2DTranspose(256,5,5))
model.add(Conv2DTranspose(256,5,5))
print(model.summary())


# In[22]:





# In[ ]:




