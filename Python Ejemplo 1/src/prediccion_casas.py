import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Etapa 1: Cargar y explorar datos
print("Etapa 1: Cargar y explorar datos")
# Cargamos el dataset California Housing de scikit-learn
housing = fetch_california_housing()
# Convertimos a DataFrame de pandas para facilitar el manejo
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['MedHouseVal'] = housing.target  # Agregamos la variable objetivo

print("Shape del dataset:", data.shape)
print("\nDescripción estadística:")
print(data.describe())
print("\nCorrelaciones con el precio medio de la casa:")
print(data.corr()['MedHouseVal'].sort_values(ascending=False))

# Análisis: El dataset tiene 20640 muestras y 9 características.
# La correlación más alta con MedHouseVal es MedInc (0.69), lo que indica que el ingreso medio es un fuerte predictor.

# Etapa 2: Separar variables predictoras
print("\nEtapa 2: Separar variables predictoras")
X = data.drop('MedHouseVal', axis=1)  # Variables predictoras
y = data['MedHouseVal']  # Variable objetivo

# Etapa 3: Entrenar modelo
print("\nEtapa 3: Entrenar modelo")
# Dividimos en conjunto de entrenamiento y prueba (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Usamos Regresión Lineal como modelo de aprendizaje supervisado
model = LinearRegression()
model.fit(X_train, y_train)

# Etapa 4: Evaluar resultados
print("\nEtapa 4: Evaluar resultados")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Análisis: El MSE es 0.556, indicando el error cuadrático medio.
# R² de 0.576 significa que el modelo explica el 57.6% de la variabilidad en los precios.

# Etapa 5: Visualizar resultados
print("\nEtapa 5: Visualizar resultados")
# Gráfico de barras para una muestra de 12 predicciones
sample_size = 12
indices = np.random.choice(len(y_test), sample_size, replace=False)
y_test_sample = y_test.iloc[indices]
y_pred_sample = y_pred[indices]

x = np.arange(sample_size)
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, y_test_sample, width, label='Valores Reales', color='skyblue')
bars2 = ax.bar(x + width/2, y_pred_sample, width, label='Valores Predichos', color='salmon')

ax.set_xlabel('Muestras')
ax.set_ylabel('Precio Medio de la Casa (en cientos de miles)')
ax.set_title('Comparación de Valores Reales vs Predichos (Muestra de 12)')
ax.set_xticks(x)
ax.set_xticklabels([f'M{i+1}' for i in range(sample_size)])
ax.legend()

plt.tight_layout()
plt.savefig('Python Ejemplo 1/results/grafico_comparacion.png')  # Guardamos el gráfico
plt.show()
