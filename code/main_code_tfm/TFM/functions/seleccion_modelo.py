import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
import warnings
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
from pandas.errors import SettingWithCopyWarning
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def obtener_importancia_variables_rf(
    X,
    y,
    tipo="regresion",
    n_estimadores=100,
    metodo="importancia_nativa",
    top_variables=None,
    mostrar_subplots=False,
    columnas_subplots=4,
    umbral_importancia=0.8,
    num_variables=None,
    variables_forzadas=None,
):
    """
    Obtiene la importancia de variables usando Random Forest para problemas multioutput.
    Ajusta un modelo independiente para cada variable objetivo.

    Parámetros:
    - X: DataFrame de características
    - y: DataFrame de variables objetivo
    - tipo: 'regresion' o 'clasificacion'
    - n_estimadores: número de árboles en el Random Forest
    - metodo: 'importancia_nativa' o 'permutacion'
    - top_variables: número de variables top a mostrar
    - mostrar_subplots: Si es True, muestra un subplot por cada variable objetivo
    - columnas_subplots: Número de columnas para los subplots
    - umbral_importancia: porcentaje de importancia acumulada para seleccionar variables (por defecto 0.8)
    - num_variables: número máximo de variables a seleccionar (se seleccionan hasta que se alcanza el umbral o hasta llegar a este número)
    - variables_forzadas: Lista de nombres de variables que deben ser seleccionadas independientemente de los umbrales

    Retorna:
    - DataFrame con la importancia promedio de las variables
    - Diccionario con importancia de variables por cada variable objetivo
    - Diccionario con las variables que cubren el umbral de importancia
    """
    importancias_por_variable = {}
    importancia_promedio = np.zeros(X.shape[1])
    variables_importancia_umbral = {}

    # Convertir variables_forzadas en un conjunto para facilitar la búsqueda
    if variables_forzadas is not None:
        variables_forzadas = set(variables_forzadas)

    # Iterar sobre las variables objetivo y ajustar un modelo independiente
    for col in y.columns:
        # Seleccionar el modelo apropiado
        if tipo == "regresion":
            modelo = RandomForestRegressor(n_estimators=n_estimadores, random_state=42)
        else:
            modelo = RandomForestClassifier(n_estimators=n_estimadores, random_state=42)

        # Ajustar el modelo para la variable objetivo actual
        modelo.fit(X, y[col])

        # Calcular importancia de variables
        if metodo == "importancia_nativa":
            importancias_por_variable[col] = modelo.feature_importances_
        elif metodo == "permutacion":
            result = permutation_importance(
                modelo, X, y[col], n_repeats=10, random_state=42
            )
            importancias_por_variable[col] = result.importances_mean
        else:
            raise ValueError("Método debe ser 'importancia_nativa' o 'permutacion'")

        # Acumular las importancias para calcular el promedio
        importancia_promedio += importancias_por_variable[col]

        # Calcular variables que cubren el umbral de importancia (80%)
        importancia_acumulada = np.cumsum(np.sort(importancias_por_variable[col])[::-1])
        total_importancia = importancia_acumulada[-1]
        variables_seleccionadas = X.columns[
            np.argsort(importancias_por_variable[col])[::-1]
        ]

        # Filtrar las variables que cubren el umbral o hasta alcanzar el número máximo de variables
        if num_variables:
            variables_importancia_umbral[col] = variables_seleccionadas[:num_variables]
        else:
            variables_importancia_umbral[col] = variables_seleccionadas[
                importancia_acumulada / total_importancia <= umbral_importancia
            ]

        # Asegurar que las variables forzadas estén en las seleccionadas
        if variables_forzadas:
            for var in variables_forzadas:
                if var in X.columns:
                    if var not in variables_importancia_umbral[col]:
                        print("Variable añadida para la columna: ", col)
                        variables_importancia_umbral[col] = np.append(
                            variables_importancia_umbral[col], var
                        )

    # Calcular el promedio de las importancias
    importancia_promedio /= len(y.columns)

    # Crear DataFrame de importancia promedio
    df_importancia = pd.DataFrame(
        {"Variable": X.columns, "Importancia": importancia_promedio}
    ).sort_values("Importancia", ascending=False)

    # Filtrar top variables si se especifica
    if top_variables:
        df_importancia = df_importancia.head(top_variables)

    # Graficar importancia promedio
    plt.figure(figsize=(10, 6))
    plt.bar(df_importancia["Variable"], df_importancia["Importancia"])
    plt.title("Importancia de Variables (Promedio)")
    plt.xlabel("Variables")
    plt.ylabel("Importancia")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # Subplots por variable objetivo si se requiere
    if mostrar_subplots:
        n_objetivos = len(y.columns)
        filas_subplots = math.ceil(
            n_objetivos / columnas_subplots
        )  # Calcular filas necesarias
        fig, axes = plt.subplots(
            filas_subplots, columnas_subplots, figsize=(15, 4 * filas_subplots)
        )
        axes = axes.flatten()  # Asegurarse de que `axes` sea una lista plana
        for i, col in enumerate(y.columns):
            importancias = importancias_por_variable[col]
            sorted_indices = np.argsort(importancias)[::-1]
            axes[i].bar(X.columns[sorted_indices], importancias[sorted_indices])
            axes[i].set_title(f"Importancia para {col}")
            axes[i].set_xticks(range(len(X.columns)))
            axes[i].set_xticklabels(X.columns[sorted_indices], rotation=45, ha="right")
            axes[i].set_ylabel("Importancia")

        # Ocultar ejes sobrantes
        for j in range(len(y.columns), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    return df_importancia, importancias_por_variable, variables_importancia_umbral


def obtener_importancia_variables(X, umbral_varianza=0.95):
    """
    Obtiene la importancia de las variables originales usando los loadings
    de los componentes principales que explican un umbral de varianza

    Parámetros:
    - X: DataFrame de datos con características
    - umbral_varianza: Porcentaje de varianza acumulada a preservar (defecto 95%)

    Retorna:
    - DataFrame con la importancia de cada variable original
    """
    # Estandarizar los datos
    scaler = StandardScaler()
    X_escalado = scaler.fit_transform(X)

    # Realizar PCA inicial para determinar número de componentes
    pca = PCA()
    pca.fit(X_escalado)

    # Calcular varianza acumulada
    varianza_acumulada = np.cumsum(pca.explained_variance_ratio_)

    # Determinar número de componentes a preservar
    num_componentes = np.argmax(varianza_acumulada >= umbral_varianza) + 1

    print(f"Número de componentes seleccionados: {num_componentes}")
    print("Varianza explicada por estos componentes:")
    varianza_componentes = pca.explained_variance_ratio_[:num_componentes]
    for i, var in enumerate(varianza_componentes):
        print(f"Componente {i+1}: {var*100:.2f}%")
    print(f"Varianza acumulada: {varianza_acumulada[num_componentes-1]*100:.2f}%")

    # Obtener los loadings de los componentes seleccionados
    loadings = pca.components_[:num_componentes]

    # Calcular la importancia de las variables
    # Usar la suma de los cuadrados de los loadings para cada variable
    importancia_variables = np.sum(loadings**2, axis=0)

    # Normalizar la importancia para que sume 1
    importancia_variables = importancia_variables / np.sum(importancia_variables)

    # Crear DataFrame con la importancia de las variables
    df_importancia = pd.DataFrame(
        {"Variable": X.columns, "Importancia": importancia_variables}
    ).sort_values("Importancia", ascending=False)

    # Graficar la importancia de las variables
    plt.figure(figsize=(10, 6))
    plt.bar(df_importancia["Variable"], df_importancia["Importancia"])
    plt.title(
        f"Importancia de Variables Originales\n(Componentes explicando {varianza_acumulada[num_componentes-1]*100:.2f}% de varianza)"
    )
    plt.xlabel("Variables")
    plt.ylabel("Importancia")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # print("\nImportancia de las Variables:")
    # print(df_importancia)

    return df_importancia


def plot_importancia(best_models, X, y, variables_importantes):
    variable_interes = "INC_SMI_REAL"  # Sustituir por el nombre de la variable

    # Diccionario para almacenar los resultados
    importances_list = []

    # Calcular la importancia para cada modelo
    for target_variable, model in best_models.items():
        # Obtener la importancia de las variables para el modelo
        result = permutation_importance(
            model,
            X[variables_importantes[target_variable]],
            y[target_variable],
            n_repeats=30,
            random_state=42,
        )

        # Buscar la importancia de la variable de interés
        if variable_interes in X[variables_importantes[target_variable]].columns:
            temp_df = pd.DataFrame(
                {
                    "Variable": [variable_interes],
                    "Importancia": [
                        result.importances_mean[
                            X[variables_importantes[target_variable]].columns.get_loc(
                                variable_interes
                            )
                        ]
                    ],
                    "Modelo": [target_variable],
                }
            )
            # Añadir el resultado a la lista
            importances_list.append(temp_df)

    # Combinar todos los DataFrames en uno solo
    importances_df = pd.concat(importances_list, ignore_index=True)

    # Crear un gráfico de barras
    plt.figure(figsize=(8, 6))
    sns.barplot(x="Importancia", y="Modelo", data=importances_df, palette="viridis")

    # Títulos y etiquetas
    plt.title(
        f'Importancia de la Variable "{variable_interes}" en las Diferentes Variables Objetivo'
    )
    plt.xlabel("Importancia Promedio")
    plt.ylabel("Modelo")

    # Mostrar el gráfico
    plt.show()
