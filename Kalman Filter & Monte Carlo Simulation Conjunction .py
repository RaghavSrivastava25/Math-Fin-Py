#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from numpy.linalg import inv
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


# A random market model
np.random.seed(0)
n = 100
t = np.arange(n)
m = 0.1 * t + np.random.normal(size=n)
v = 0.01 * t + np.random.normal(size=n)


# In[ ]:


# Initialize the Kalman filter parameters
F = np.array([[1]])
H = np.array([[1]])
Q = np.array([[0.01]])
R = np.array([[0.1]])


# In[ ]:


# Initialize the Kalman filter
x = np.array([[m[0]]])
P = np.array([[1]])


# In[ ]:


# Run the Kalman filter and Monte Carlo simulation
N = 10000

portfolio = np.zeros(N)
for i in range(1, n):
    # Predict the next state using the Kalman filter
    x_pred = F.dot(x)
    P_pred = F.dot(P).dot(F.T) + Q

    # Update the state estimate based on the new measurement
    y = np.array([[m[i]]])
    K = P_pred.dot(H.T).dot(inv(H.dot(P_pred).dot(H.T) + R))
    x = x_pred + K.dot(y - H.dot(x_pred))
    P = (np.eye(1) - K.dot(H)).dot(P_pred)

    # Generate Monte Carlo scenarios for the next time step
    var = abs(v[i])
    m_sim = np.random.normal(x[0, 0], var, N)
    portfolio += m_sim - m[i-1]


# In[3]:


# Plot the results
plt.plot(t, m, label='Market')
plt.plot(t[1:], portfolio[:99], label='Portfolio')
plt.legend()
plt.show()


# In[ ]:




