# Predicción de Precios de Casas - Aprendizaje Supervisado

Este proyecto implementa un modelo de aprendizaje supervisado para predecir los precios medios de las casas en California utilizando el dataset California Housing de scikit-learn.

## Descripción del Proyecto

El objetivo es demostrar las etapas fundamentales del aprendizaje supervisado aplicado a un problema de regresión: predecir el precio medio de las casas basado en características demográficas y geográficas.

## Librerías Utilizadas

- **pandas**: Manipulación y análisis de datos
- **numpy**: Operaciones numéricas
- **matplotlib**: Visualización de datos
- **scikit-learn**: Algoritmos de machine learning

## Etapas del Proyecto

### 1. Cargar y Explorar Datos
- Dataset: California Housing (20,640 muestras, 9 características)
- Características incluidas:
  - MedInc: Ingreso medio
  - HouseAge: Edad media de la casa
  - AveRooms: Número promedio de habitaciones
  - AveBedrms: Número promedio de dormitorios
  - Population: Población del bloque
  - AveOccup: Ocupación promedio
  - Latitude: Latitud
  - Longitude: Longitud
- Variable objetivo: MedHouseVal (Precio medio en cientos de miles de dólares)

### 2. Separar Variables Predictoras
- X: Características predictoras
- y: Variable objetivo (MedHouseVal)

### 3. Entrenar Modelo
- Modelo: Regresión Lineal
- División de datos: 80% entrenamiento, 20% prueba
- Random state: 42 para reproducibilidad

### 4. Evaluar Resultados
- **Mean Squared Error (MSE)**: 0.556
- **R² Score**: 0.576 (57.6% de variabilidad explicada)

### 5. Visualizar Resultados
- Gráfico de barras comparando valores reales vs predichos (muestra de 12 ejemplos)
- Guardado en `Python Ejemplo 1/results/grafico_comparacion.png`

## Estructura del Proyecto

```
Python Ejemplo 1/
├── docs/
│   └── analisis.md          # Análisis detallado del proyecto
├── results/
│   └── grafico_comparacion.png  # Gráfico de comparación
└── src/
    └── prediccion_casas.py  # Script principal
```

## Cómo Ejecutar

1. Asegúrate de tener instaladas las librerías requeridas:
   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```

2. Ejecuta el script:
   ```bash
   python Python Ejemplo 1/src/prediccion_casas.py
   ```

## Análisis de Resultados

- **Correlaciones principales**: MedInc (0.69) es el predictor más fuerte
- **Rendimiento del modelo**: Explica el 57.6% de la variabilidad en precios
- **Limitaciones**: Modelo lineal simple; podría mejorarse con algoritmos más complejos

## Mejoras Futuras

- Implementar modelos más avanzados (Random Forest, Gradient Boosting)
- Ingeniería de características
- Validación cruzada
- Análisis de residuos

## Autor

Proyecto desarrollado como parte del taller de aprendizaje supervisado.
