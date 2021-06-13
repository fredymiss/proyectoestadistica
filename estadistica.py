import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

import matplotlib.pyplot as plt
file = 'link del archivo puede ser el path local o un link web'
dt = pd.read_csv(file)
dt.head()
#dt.plot(kind='scatter', x='x1', y='y');
X = dt["x1"]                   
X = sm.add_constant(X.values)  
y = dt["y"]                    
print(X[0:5])
print('\n')
print(y[0:5])
mod1 = sm.OLS(y, X).fit() 
predictions = mod1.predict(X)
mod1.summary()
mod1.summary()
mod2 = smf.ols('y ~ x1', data=dt).fit()
print(mod2.summary())
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(dt["x1"] , dt["y"] , 'o', label="Datos")
ax.plot(dt["x1"], mod2.fittedvalues, 'r--.', label="Ajustado")
legend = ax.legend(loc="best")
plt.xlabel('Cantidad de ')
plt.ylabel('Viarable ')
plt.title('Diagrama de dispersión con la recta del modelo ajustado');