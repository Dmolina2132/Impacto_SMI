import pandas as pd
import sklearn as sk
import numpy as np
import math
import statsmodels.api as sm
import warnings
import utils as u
from pandas.errors import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, GridSearchCV


def evaluacion_modelo_simple(X, y, variables_importantes, model):
    """
    Evalúa el rendimiento de un modelo con respecto a un conjunto de variables objetivo,
    realizando una validación cruzada con KFold.

    Parameters
    ----------
    X : pandas.DataFrame
        Conjunto de datos con las variables predictoras.
    y : pandas.DataFrame
        Conjunto de datos con las variables objetivo.
    variables_importantes : dict
        Diccionario con las variables predictoras y su correspondiente variable objetivo.
    model : sklearn.Model
        Modelo a evaluar.

    Returns
    -------
    pandas.DataFrame
        DataFrame con los resultados de la evaluación del modelo. Cada fila representa una
        variable objetivo y contiene el promedio de las puntuaciones R² obtenidas en la
        validación cruzada.
    """
    results = []
    # Realizar la validación cruzada para cada variable objetivo
    for target_variable, predictors in variables_importantes.items():
        # Extraer las columnas del DataFrame correspondientes a las variables predictoras
        X_var = X[predictors]  # Variables predictoras
        y_var = y[target_variable]  # Variable objetivo

        # Configurar KFold (opcional, puedes elegir otros parámetros de partición)
        kf = KFold(
            n_splits=5, shuffle=True, random_state=42
        )  # 5 pliegues de cross-validation

        # Realizar el cross-validation y obtener la puntuación (por defecto, R^2)
        cv_scores = cross_val_score(
            model, X_var, y_var, cv=kf, scoring="r2"
        )  # Cambiar 'r2' si necesitas otro tipo de métrica

        # Guardar los resultados para la variable objetivo actual
        results.append(
            {
                "Variable Objetivo": target_variable,
                "Mean R²": cv_scores.mean(),  # Promedio de las puntuaciones R^2
            }
        )
    results_df = pd.DataFrame(results)
    return results_df


def evaluacion_modelo(X, y, variables_importantes, model, param_grid):
    """
    Evalúa el rendimiento de un modelo con respecto a un conjunto de variables objetivo, realizando una búsqueda de hiperparámetros con GridSearchCV.

    Parameters
    ----------
    X : pandas.DataFrame
        Conjunto de datos con las variables predictoras.
    y : pandas.DataFrame
        Conjunto de datos con las variables objetivo.
    variables_importantes : dict
        Diccionario con las variables predictoras y su correspondiente variable objetivo.
    model : sklearn.Model
        Modelo a evaluar.
    param_grid : dict
        Diccionario con los parámetros a explorar en la búsqueda de hiperparámetros.

    Returns
    -------
    pandas.DataFrame
        DataFrame con los resultados de la evaluación del modelo. Cada fila representa una variable objetivo y contiene el mejor R² obtenido.
    dict
        Diccionario con los mejores parámetros para cada variable objetivo.
    """
    results = []
    best_params_dict = {}

    # Realizar la búsqueda de hiperparámetros con GridSearchCV para cada variable objetivo
    for target_variable, predictors in variables_importantes.items():
        # Extraer las columnas del DataFrame correspondientes a las variables predictoras
        X_var = X[predictors]  # Variables predictoras
        y_var = y[target_variable]  # Variable objetivo

        # Configurar KFold (5 pliegues de cross-validation)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # Realizar la búsqueda de hiperparámetros con GridSearchCV
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=kf,
            scoring="r2",
            n_jobs=-1,
            verbose=0,
        )

        # Ajustar el modelo con los datos
        grid_search.fit(X_var, y_var)

        # Obtener el mejor conjunto de parámetros
        best_params = grid_search.best_params_

        # Obtener el mejor R² para la mejor combinación de parámetros
        best_r2 = grid_search.best_score_

        # Almacenar el resultado de R² y el mejor conjunto de parámetros
        results.append({"Variable Objetivo": target_variable, "Best R²": best_r2})

        # Guardar el mejor conjunto de parámetros para esta variable objetivo
        best_params_dict[target_variable] = best_params

    # Convertir los resultados en un DataFrame
    results_df = pd.DataFrame(results)
    return results_df, best_params_dict
