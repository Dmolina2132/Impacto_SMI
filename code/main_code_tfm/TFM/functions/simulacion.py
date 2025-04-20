from scipy.optimize import minimize
import pandas as pd
import numpy as np
import warnings
from pandas.errors import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def model_prediction(inc_smi, data, best_models, variables_importantes):
    # Creamos la prediccion para todos los valores
    results = {}
    data_copy = data.copy()
    data_copy["INC_SMI_REAL"] = inc_smi
    for target_val, best_model in best_models.items():
        results[target_val] = best_model.predict(
            data_copy[variables_importantes[target_val]]
        )[0]
    return results


def increase_vars(inc_values, data):
    # Recorremos los valores incrementales y seleccionamos la variable para aumentar el valor correspondiente

    for col in data.columns:
        for inc in inc_values:
            if col in inc:
                if col != "CARENCIA":
                    data[col] *= 1 + inc_values[inc]
                else:
                    data[col] += inc_values[inc]
                break
    return data


def simulacion_smi(
    min_inc,
    max_inc,
    df,
    fun,
    best_models,
    variables_importantes,
    pasos=5,
):
    """
    Realiza una simulacion de aumento del salario minimo, seleccionando el valor que maximice la función dada
    en pasos de 150 valores y para cada paso aplica la prediccion de los modelos para aumentar
    las variables correspondientes una vez seleccionado el incremento óptimo. Al final devuelve un DataFrame con la evolucion de las variables
    en cada paso. Se ha elegido recorrer directamente los valores en lugar de usar modulos como scipy
    pues la irregularidad de la funcion da problemas con la optimizacion.

    Parameters
    ----------
    min_inc : float
        Valor minimo de aumento del salario minimo permitido
    max_inc : float
        Valor maximo de aumento del salario minimo permitido
    df : pd.DataFrame
        DataFrame con los datos base
    fun : funcion
        Funcion que se busca maximizar
    best_models : dict
        Diccionario con los modelos de prediccion para cada variable objetivo
    variables_importantes : dict
        Diccionario con las variables importantes para cada variable objetivo
    pasos : int
        Numero de pasos que se realizan en la simulacion

    Returns
    -------
    pd.DataFrame
        DataFrame con la evolucion de las variables en cada paso
    """
    evolution = []
    df_temp = df.copy()
    for step in range(pasos):
        best_value = -100000000
        best_inc = 0
        for inc in np.linspace(min_inc, max_inc, 150):
            val = fun(inc, df_temp, best_models, variables_importantes)
            if val > best_value:
                best_value = val
                best_inc = inc
        # Apply the best increase
        df_temp["INC_SMI_REAL"] = best_inc
        evolution.append(df_temp.copy())
        increases = model_prediction(
            best_inc, df_temp, best_models, variables_importantes
        )

        df_temp = increase_vars(increases, df_temp)
    evol_df = pd.concat(evolution)
    return evol_df
