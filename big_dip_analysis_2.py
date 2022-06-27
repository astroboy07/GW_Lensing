import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def magnification(y, ML, lens = 'sis'):
    
        if lens == 'pm':
            mu_plus = np.abs(0.5 + (y ** 2 + 2) / (2 * y * (y ** 2 + 4) ** 0.5))
            mu_minus = np.abs(0.5 - (y ** 2 + 2) / (2 * y * (y ** 2 + 4) ** 0.5))
        
        elif lens == 'sis':
            mu_plus = np.abs(1 + 1 / y)
            mu_minus = np.abs(-1 + 1 / y)
        
        return mu_minus / mu_plus, mu_plus, mu_minus

def time_del(y, ML, lens = 'sis'):

    if lens == 'pm':
        first_term = (y * (y ** 2 + 4) ** 0.5) / 2
        second_term = np.log(((y ** 2 + 4) ** 0.5 + y) / ((y ** 2 + 4) ** 0.5 - y))
        tds = 4 * ML * (first_term + second_term)

    elif lens == 'sis':
        tds = 8 * ML * y

    return tds

df = pd.read_csv("/Users/saifali/Desktop/gwlensing/mismatch_mesh_pm.csv")
y_mesh_pm = np.array(df['y_pm']).reshape(60, 60)
ML_mesh_pm = np.array(df['ML_pm']).reshape(60, 60)
mismatch_mesh_pm = np.array(df['mismatch_pm']).reshape(60, 60)

I_mesh_pm = magnification(y_mesh_pm, ML_mesh_pm, lens = 'pm')[0]
td_mesh_pm = time_del(y_mesh_pm, ML_mesh_pm, lens = 'pm')

ind = 10
print(I_mesh_pm[:, ind][0])
plt.scatter(td_mesh_pm[:, ind], mismatch_mesh_pm[:, ind])
plt.show()