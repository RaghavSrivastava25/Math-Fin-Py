#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


# Load financial data
df = pd.read_csv('financial_data.csv', index_col=0)

# Perform SVD
U, S, VT = np.linalg.svd(df)

# Perform EVD
eigvals, eigvecs = np.linalg.eig(np.cov(df.T))


# In[ ]:


# Print the singular values and eigenvalues
print("Singular values:\n", S)
print("Eigenvalues:\n", eigvals)

# Print the first principal component
print("First principal component (SVD):\n", VT[0])
print("First principal component (EVD):\n", eigvecs[:,0])

