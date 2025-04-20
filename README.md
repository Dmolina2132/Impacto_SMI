# Predicción de efectos del salario mínimo mediante algoritmos de aprendizaje automático

Este repositorio contiene el código y los recursos para el Trabajo de Fin de Máster (TFM) titulado "Predicción de efectos del salario mínimo mediante algoritmos de aprendizaje automático", realizado por Diego Molina González.

## 1. Introducción

El estudio analiza los efectos del salario mínimo en las diferentes regiones de España, considerando sus particularidades económicas. Se utilizan algoritmos de aprendizaje automático para predecir los efectos económicos de los incrementos del salario mínimo.

## 2. Objetivos

Los objetivos principales del estudio son:

* Determinar los efectos económicos generales de un incremento del salario mínimo en cada comunidad autónoma.
* Identificar un punto y ritmo óptimo de subida del salario mínimo que maximice sus beneficios.

## 3. Base de Datos

Los datos utilizados provienen principalmente del Instituto Nacional de Estadística (INE), la Agencia Tributaria Española y el Ministerio de Seguridad Social. Los datasets empleados incluyen:

* Encuestas de estructura salarial: salarios por hora, salario medio mensual, desigualdad (índice de Gini, índice S80/S20).
* Índices de precios de consumo (IPC).
* Encuesta de presupuestos familiares: gasto medio de los hogares.
* Estadísticas de movilidad nacional y geografía: número de parados.
* Encuesta de condiciones de vida: tasa de riesgo de pobreza, carencia material.
* Estadística Estructural de Empresas: número de empresas, flujo de altas y bajas.
* Encuesta de Población Activa: número de parados y ocupados, edad y tipo de trabajo.

## 4. Análisis y Modelos

El análisis y los modelos se implementaron en Python, utilizando las librerías:

* `pandas`: para el análisis de datos.
* `sklearn`: para la creación y testeo de modelos de aprendizaje automático.
* `matplotlib` y `seaborn`: para la visualización de datos.

Se incluyen los siguientes ficheros con funciones personalizadas:

* `plots.py`: Funciones para crear gráficos.
* `data_format.py`: Funciones para el formateo de datos.
* `seleccion_modelo.py`: Funciones para la selección de variables y modelos.
* `evaluacion_modelo.py`: Funciones para la evaluación de modelos (e.g., grid search).
* `simulacion.py`: Funciones para realizar simulaciones de incrementos del salario mínimo.

## 5. Análisis Descriptivo

Se realiza un análisis descriptivo de las variables clave, incluyendo:

* Salario Mínimo Interprofesional (SMI) y su evolución.
* Índice de Precios de Consumo (IPC) y su variación.
* Gasto de los hogares, tanto el gasto total como el gasto en bienes básicos.
* Salarios, analizando su evolución por ocupación, jornada, sector y rango de SMI.
* Empleo, examinando el número de ocupados y su distribución por tipo de jornada.

## 6. Correlaciones y Selección de Variables

Se realiza un análisis de correlaciones para identificar las variables más relevantes y reducir la dimensionalidad del problema. Se emplean técnicas como el Análisis de Componentes Principales (PCA) y la importancia de variables en Random Forest para la selección.

## 7. Modelado

Se implementan modelos de aprendizaje automático para predecir los efectos del salario mínimo en diversas variables económicas. Se utilizan modelos como Random Forest Regressor, Gradient Boosting Regressor, Linear Regression, Lasso, Decision Tree Regressor y SVR. Se realiza una optimización de hiperparámetros mediante Grid Search y se evalúa el rendimiento de los modelos.

## 8. Simulación de Incrementos del Salario Mínimo

Se desarrollan simulaciones para predecir los efectos de diferentes escenarios de incrementos del salario mínimo en las variables económicas de interés.

## 9. Conclusiones

El estudio proporciona información valiosa para la formulación de políticas laborales, identificando estrategias para maximizar los beneficios del salario mínimo y minimizar sus posibles efectos adversos.

## Contacto

Diego Molina González - diegomolinaglez@gmail.com
