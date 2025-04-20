# Predicción de efectos del salario mínimo mediante algoritmos de aprendizaje automático

Este repositorio contiene el código y los recursos para el Trabajo de Fin de Máster (TFM) titulado "Predicción de efectos del salario mínimo mediante algoritmos de aprendizaje automático", realizado por Diego Molina González. [cite: 304]

## 1. Introducción

El estudio analiza los efectos del salario mínimo en las diferentes regiones de España, considerando sus particularidades económicas. [cite: 308, 309, 310, 311] Se utilizan algoritmos de aprendizaje automático para predecir los efectos económicos de los incrementos del salario mínimo. [cite: 304, 308, 309, 310, 311]

## 2. Objetivos

Los objetivos principales del estudio son:

* Determinar los efectos económicos generales de un incremento del salario mínimo en cada comunidad autónoma. [cite: 309]
   
* Identificar un punto y ritmo óptimo de subida del salario mínimo que maximice sus beneficios. [cite: 310]

## 3. Base de Datos

Los datos utilizados provienen principalmente del Instituto Nacional de Estadística (INE), la Agencia Tributaria Española y el Ministerio de Seguridad Social. [cite: 312] Los datasets empleados incluyen:

* Encuestas de estructura salarial: salarios por hora, salario medio mensual, desigualdad (índice de Gini, índice S80/S20). [cite: 313, 314, 315]
   
* Índices de precios de consumo (IPC). [cite: 316]
   
* Encuesta de presupuestos familiares: gasto medio de los hogares. [cite: 317]
   
* Estadísticas de movilidad nacional y geografía: número de parados. [cite: 318]
   
* Encuesta de condiciones de vida: tasa de riesgo de pobreza, carencia material. [cite: 319]
   
* Estadística Estructural de Empresas: número de empresas, flujo de altas y bajas. [cite: 19]
   
* Encuesta de Población Activa: número de parados y ocupados, edad y tipo de trabajo. [cite: 320, 321, 322]

## 4. Análisis y Modelos

El análisis y los modelos se implementaron en Python, utilizando las librerías:

* `pandas`: para el análisis de datos. [cite: 323]
   
* `sklearn`: para la creación y testeo de modelos de aprendizaje automático. [cite: 323]
   
* `matplotlib` y `seaborn`: para la visualización de datos. [cite: 324]

Se incluyen los siguientes ficheros con funciones personalizadas:

* `plots.py`: Funciones para crear gráficos. [cite: 325, 326, 327, 328, 329, 330]
   
* `data_format.py`: Funciones para el formateo de datos. [cite: 325, 326, 327, 328, 329, 330]
   
* `seleccion_modelo.py`: Funciones para la selección de variables y modelos. [cite: 325, 326, 327, 328, 329, 330]
   
* `evaluacion_modelo.py`: Funciones para la evaluación de modelos (e.g., grid search). [cite: 325, 326, 327, 328, 329, 330]

* `simulacion.py`: Funciones para realizar simulaciones de incrementos del salario mínimo. [cite: 325, 326, 327, 328, 329, 330]

## 5. Análisis Descriptivo

Se realiza un análisis descriptivo de las variables clave, incluyendo:

* Salario Mínimo Interprofesional (SMI) y su evolución. [cite: 333, 334, 335, 336, 337, 338, 339, 340]
   
* Índice de Precios de Consumo (IPC) y su variación. [cite: 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359]
   
* Gasto de los hogares, tanto el gasto total como el gasto en bienes básicos. [cite: 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389]
   
* Salarios, analizando su evolución por ocupación, jornada, sector y rango de SMI. [cite: 384, 385, 386, 387, 388, 389]

* Empleo, examinando el número de ocupados y su distribución por tipo de jornada.

## 6. Correlaciones y Selección de Variables

Se realiza un análisis de correlaciones para identificar las variables más relevantes y reducir la dimensionalidad del problema. [cite: 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192] Se emplean técnicas como el Análisis de Componentes Principales (PCA) y la importancia de variables en Random Forest para la selección. [cite: 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192]

## 7. Modelado

Se implementan modelos de aprendizaje automático para predecir los efectos del salario mínimo en diversas variables económicas. [cite: 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303] Se utilizan modelos como Random Forest Regressor, Gradient Boosting Regressor, Linear Regression, Lasso, Decision Tree Regressor y SVR. [cite: 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303] Se realiza una optimización de hiperparámetros mediante Grid Search y se evalúa el rendimiento de los modelos. [cite: 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303]

## 8. Simulación de Incrementos del Salario Mínimo

Se desarrollan simulaciones para predecir los efectos de diferentes escenarios de incrementos del salario mínimo en las variables económicas de interés. [cite: 325, 326, 327, 328, 329, 330]

## 9. Conclusiones

El estudio proporciona información valiosa para la formulación de políticas laborales, identificando estrategias para maximizar los beneficios del salario mínimo y minimizar sus posibles efectos adversos. [cite: 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303]

## Contacto

Diego Molina González - diegomolinaglez@gmail.com

