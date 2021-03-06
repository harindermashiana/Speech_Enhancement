# -*- coding: utf-8 -*-
"""1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1faVvXpCEjGMOvileQPz52L1iBdO2LDf1
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import IPython.display as ipd

from google.colab import drive
drive.mount('/content/drive')

noisy_name=[]
for filename in os.listdir(r"/content/drive/My Drive/mproject/train/noisy_trainset"):
  noisy_name.append(filename)
noisy_name.sort()

D1=[] #for frequency
D2=[] #for phase
i=0
for f in clean_name:
 if(i<400):
  S1=[]
  S2=[]
  wav, sr=librosa.load('/content/drive/My Drive/mproject/train/noisy_trainset/'+f,duration=2)
  if((wav.shape[0]/sr) != 2.0): 
          continue
  inter=librosa.stft(wav)
  D1_=np.abs(inter)
  D2_=np.angle(inter)
  S1.append(D1_)
  S2.append(D2_)
  D1.append(S1)
  D2.append(S2)
  i=i+1
  print(i)

clean_name=[]
for filename in os.listdir(r"/content/drive/My Drive/mproject/train/clean_trainset"):
  clean_name.append(filename)
clean_name.sort()

print(len(clean_name))
print(len(noisy_name))

Xf=np.array(D1)
Xp=np.array(D2)
Xf=np.rollaxis(Xf,1,4)
Xp=np.rollaxis(Xp,1,4)
print(Xf.shape)

D11=[] #for frequency
D22=[] #for phase
i=0
for f in clean_name:
 if(i<400):
  S1=[]
  S2=[]
  wav, sr=librosa.load('/content/drive/My Drive/mproject/train/clean_trainset/'+f,duration=2)
  if((wav.shape[0]/sr) != 2.0): continue
  inter=librosa.stft(wav)
  D1_=np.abs(inter)
  D2_=np.angle(inter)
  S1.append(D1_)
  S2.append(D2_)
  D11.append(S1)
  D22.append(S2)
  i=i+1
  print(f)

Yf=np.array(D11)
Yp=np.array(D22)
Yf=np.rollaxis(Yf,1,4)
Yp=np.rollaxis(Yp,1,4)
print(Yf.shape)

from tempfile import TemporaryFile

np.save('trainfrequency', Xf)
np.save('trainphase', Xp)
np.save('testfrequency', Yf)
np.save('testphase', Yp)



