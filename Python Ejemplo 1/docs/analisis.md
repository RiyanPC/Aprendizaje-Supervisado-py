# Análisis de Predicción de Precios de Casas

## Descripción del Proyecto
Este proyecto implementa un modelo de aprendizaje supervisado para predecir los precios medios de las casas en California utilizando el dataset California Housing de scikit-learn.

## Etapas Realizadas

### 1. Cargar y Explorar Datos
- **Dataset**: California Housing (20640 muestras, 9 características)
- **Características**:
  - MedInc: Ingreso medio
  - HouseAge: Edad media de la casa
  - AveRooms: Número promedio de habitaciones
  - AveBedrms: Número promedio de dormitorios
  - Population: Población del bloque
  - AveOccup: Ocupación promedio
  - Latitude: Latitud
  - Longitude: Longitud
- **Variable Objetivo**: MedHouseVal (Precio medio de la casa en cientos de miles de dólares)

**Análisis Exploratorio**:
- Estadísticas descriptivas muestran rangos variados en las características.
- Correlaciones: MedInc tiene la correlación más alta (0.69) con el precio, indicando que el ingreso es un fuerte predictor.

### 2. Separar Variables Predictoras
- X: Todas las características excepto MedHouseVal
- y: MedHouseVal

### 3. Entrenar Modelo
- Modelo: Regresión Lineal
- División: 80% entrenamiento, 20% prueba
- El modelo se ajusta a los datos de entrenamiento.

### 4. Evaluar Resultados
- **Mean Squared Error (MSE)**: 0.556
- **R² Score**: 0.576

**Interpretación**:
- MSE mide el error cuadrático medio entre predicciones y valores reales.
- R² indica que el modelo explica el 57.6% de la variabilidad en los precios de las casas.

### 5. Visualizar Resultados
- Gráfico de dispersión: Valores reales vs. predichos
- Guardado en `results/grafico_comparacion.png`

## Conclusiones
El modelo de regresión lineal proporciona una predicción razonable de los precios de las casas, con un R² de 0.58. Mejoras futuras podrían incluir:
- Uso de modelos más complejos (Random Forest, Gradient Boosting)
- Ingeniería de características
- Validación cruzada
