#!/usr/bin/env python
# coding: utf-8

# In[1]:


P = [[0.9, 0.1, 0.0],
     [0.2, 0.7, 0.1],
     [0.1, 0.2, 0.7]]


# In[ ]:


import numpy as np


# In[ ]:


# Define transition matrix
P = np.array([[0.9, 0.1, 0.0],
              [0.2, 0.7, 0.1],
              [0.1, 0.2, 0.7]])


# In[ ]:


# Compute eigenvalues and eigenvectors of transpose of P
eig_vals, eig_vecs = np.linalg.eig(P.T)

# Find index of largest eigenvalue
max_eig_idx = np.argmax(eig_vals)

# Extract eigenvector corresponding to largest eigenvalue
stationary_prob_vec = eig_vecs[:, max_eig_idx].real
stationary_prob_vec /= stationary_prob_vec.sum()


# In[ ]:


print('Stationary probability vector:', stationary_prob_vec)


# In[ ]:




