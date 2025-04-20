import pandas as pd
import sklearn as sk
import numpy as np
import math
import statsmodels.api as sm
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.errors import SettingWithCopyWarning
from matplotlib.dates import AutoDateLocator
from sklearn.inspection import permutation_importance

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def create_basic_plot(
    x,
    y,
    xlabel="",
    ylabel="",
    title="",
    xticks_rotation=0,
    style="whitegrid",
    color="dodgerblue",
    marker="o",
    figsize=(10, 6),
    save_path=None,
    label="Salario Mínimo",
):
    # Set the Seaborn style for better aesthetics
    sns.set_style(style)

    # Create the plot
    plt.figure(figsize=figsize)
    plt.plot(x, y, marker=marker, color=color, linewidth=2.5, markersize=8, label=label)

    # Add labels, title, and grid
    plt.xlabel(xlabel, fontsize=12, labelpad=10)
    plt.ylabel(ylabel, fontsize=12, labelpad=10)
    plt.title(title, fontsize=14, weight="bold", pad=15)
    plt.xticks(rotation=xticks_rotation, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(visible=True, linestyle="--", alpha=0.7)

    # Add a legend if needed
    plt.legend(loc="best", fontsize=10, frameon=True)

    # Improve layout
    plt.tight_layout()
    locator = AutoDateLocator()  # Ajusta automáticamente el número de fechas visibles
    plt.gca().xaxis.set_major_locator(locator)

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    # Show the plot
    plt.show()


def create_dual_plot(
    x,
    y1,
    y2,
    xlabel="",
    ylabel1="",
    ylabel2="",
    title="",
    xticks_rotation=0,
    style="whitegrid",
    color1="dodgerblue",
    color2="orange",
    marker1="o",
    marker2="s",
    label1="Serie 1",
    label2="Serie 2",
    figsize=(10, 6),
    secondary_y=False,
    save_path=None,
):
    # Set the Seaborn style for better aesthetics
    sns.set_style(style)

    # Create the figure and primary axis
    fig, ax1 = plt.subplots(figsize=figsize)

    # Plot the first series on the primary y-axis
    line1 = ax1.plot(
        x, y1, marker=marker1, color=color1, linewidth=2.5, markersize=8, label=label1
    )
    ax1.plot(
        x, y1, marker=marker1, color=color1, linewidth=2.5, markersize=8, label=label1
    )
    ax1.set_xlabel(xlabel, fontsize=12, labelpad=10)
    ax1.set_ylabel(ylabel1, fontsize=12, labelpad=10, color=color1)
    ax1.tick_params(axis="y")
    ax1.tick_params(axis="x", rotation=xticks_rotation)
    ax1.grid(visible=True, linestyle="--", alpha=0.7)

    if secondary_y:
        # Add a second y-axis
        ax2 = ax1.twinx()
        line2 = ax2.plot(
            x,
            y2,
            marker=marker2,
            color=color2,
            linewidth=2.5,
            markersize=8,
            label=label2,
        )
        ax2.plot(
            x,
            y2,
            marker=marker2,
            color=color2,
            linewidth=2.5,
            markersize=8,
            label=label2,
        )
        ax2.set_ylabel(ylabel2, fontsize=12, labelpad=10, color=color2)
        ax2.tick_params(axis="y")
        # ax2.legend(loc="upper right", fontsize=10, frameon=True)
    else:
        line2 = []
        # Plot the second series on the primary y-axis
        ax1.plot(
            x,
            y2,
            marker=marker2,
            color=color2,
            linewidth=2.5,
            markersize=8,
            label=label2,
        )
        # ax1.legend(loc="best", fontsize=10, frameon=True)

    # Add title and legend for the first axis
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best", fontsize=10, frameon=True)
    fig.suptitle(title, fontsize=14, weight="bold", y=1.02)

    # Adjust layout
    fig.tight_layout()

    # Adjust x-axis for date formatting if needed
    locator = AutoDateLocator()  # Automatically adjusts visible date labels
    ax1.xaxis.set_major_locator(locator)

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    # Show the plot
    plt.show()


def create_multi_category_plot(
    data,
    x_col,
    y_col,
    category_col,
    xlabel="",
    ylabel="",
    label=None,
    title="",
    xticks_rotation=0,
    style="whitegrid",
    palette="husl",
    figsize=(12, 7),
    save_path=None,
):
    if label is None:
        label = category_col
    # Set Seaborn style for better aesthetics
    sns.set_style(style)

    # Create the plot
    plt.figure(figsize=figsize)
    unique_categories = data[category_col].unique()
    colors = sns.color_palette(palette, len(unique_categories))

    for i, category in enumerate(unique_categories):
        subset = data[data[category_col] == category]
        plt.plot(
            subset[x_col],
            subset[y_col],
            # marker="o",
            linewidth=2.5,
            markersize=8,
            label=category,
            color=colors[i],
        )

    # Add labels, title, and grid
    plt.xlabel(xlabel, fontsize=12, labelpad=10)
    plt.ylabel(ylabel, fontsize=12, labelpad=10)
    plt.title(title, fontsize=14, weight="bold", pad=15)
    plt.xticks(rotation=xticks_rotation, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(visible=True, linestyle="--", alpha=0.7)

    # Add a legend
    plt.legend(title=label, fontsize=10, title_fontsize=12, loc="best", frameon=True)

    # Improve layout
    plt.tight_layout()
    locator = AutoDateLocator()  # Ajusta automáticamente el número de fechas visibles
    plt.gca().xaxis.set_major_locator(locator)
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    # Show the plot
    plt.show()


def creat_corr_matrix(df, num_var, title="Mapa de calor de correlación"):
    correlation_matrix = df[num_var].corr()

    # Configurar el tamaño del gráfico
    plt.figure(figsize=(10, 8))

    # Crear el mapa de calor con seaborn
    sns.heatmap(
        correlation_matrix,
        annot=True,  # Muestra los valores numéricos en cada celda
        fmt=".2f",  # Formato de los números (2 decimales)
        cmap="coolwarm",  # Colores (puedes usar "viridis", "magma", etc.)
        cbar=True,  # Barra de color
        square=True,
    )  # Hacer que cada celda sea cuadrada

    # Añadir título
    plt.title(title, fontsize=16)

    # Mostrar el gráfico
    plt.show()


def create_category_subplots(
    data,
    x_col,
    y_col,
    category_col,
    xlabel="",
    ylabel="",
    title="",
    xticks_rotation=45,
    style="whitegrid",
    palette="husl",
    figsize=(30, 22.5),
    n_cols=3,
    save_path=None,
):
    """
    Create individual subplots for each category in the data with adjusted margins.
    """
    import matplotlib.ticker as mticker

    # Set Seaborn style for aesthetics
    sns.set_style(style)

    # Get unique categories and assign colors
    unique_categories = data[category_col].unique()
    n_categories = len(unique_categories)
    n_rows = -(-n_categories // n_cols)  # Calculate rows (ceiling division)
    colors = sns.color_palette(palette, n_categories)

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=False, sharey=True)
    axes = axes.flatten()  # Flatten to iterate easily

    for i, category in enumerate(unique_categories):
        ax = axes[i]
        subset = data[data[category_col] == category]
        ax.plot(
            subset[x_col],
            subset[y_col],
            # marker="o",
            linewidth=1.5,
            markersize=4,
            label=category,
            color=colors[i],
        )
        ax.set_title(f"{category}", fontsize=10, weight="bold", pad=5)
        ax.grid(visible=True, linestyle="--", alpha=0.7)

        # Set x-axis ticks to show fewer values
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
        ax.tick_params(axis="x", rotation=xticks_rotation, labelsize=6)
        ax.tick_params(axis="y", labelsize=6)

    # Hide unused subplots
    for j in range(len(axes)):
        if j >= n_categories:
            axes[j].set_visible(False)

    # Set labels for shared axes
    fig.text(0.5, 0.02, xlabel, ha="center", fontsize=12)  # Adjusted position
    fig.text(
        0.04, 0.5, ylabel, va="center", rotation="vertical", fontsize=12
    )  # Adjusted position

    # Add the main title
    fig.suptitle(title, fontsize=16, weight="bold", y=0.98)

    # Adjust layout with extra margins
    fig.subplots_adjust(
        left=0.08, right=0.97, bottom=0.08, top=0.92, wspace=0.3, hspace=1
    )
    fig.tight_layout()
    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    # Show the plot
    plt.show()


# Plot para la simulación
def plot_simulacion(df_res_1, df_res_2, ccaa_1, ccaa_2, variables, n_columns=5):
    vars = df_res_1.columns[1:]

    # Calcular el número de filas necesarias
    n_rows = int(np.ceil(len(vars) / n_columns))

    # Crear el gráfico con el número adecuado de filas y columnas
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(15, 3 * n_rows), sharex=True)

    # Si sólo hay un subplot, asegurarse de que sea iterable
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    # Hacer el gráfico para cada variable
    for i, var in enumerate(vars):
        row = i // n_columns  # Determina la fila
        col = i % n_columns  # Determina la columna
        ax = axes[row, col]

        ax.plot(df_res_1["INC_SMI_REAL"], df_res_1[var], color="b", label=ccaa_1)
        ax.plot(df_res_2["INC_SMI_REAL"], df_res_2[var], color="g", label=ccaa_2)
        ax.set_title(f"{var} vs INC_SMI_REAL", fontsize=8)
        ax.set_ylabel(var)
        ax.grid(True)
        ax.legend(fontsize=7, loc="upper right")
        ax.tick_params(axis="both", labelsize=6.5)

    # Ajustar el layout para que no se solapen los títulos y las etiquetas
    plt.tight_layout()

    # Eliminar subgráficos vacíos
    for i in range(len(variables), n_rows * n_columns):
        fig.delaxes(axes.flatten()[i])

    # Mostrar el gráfico
    plt.show()


def plot_importancia_univariable(
    best_models, X, y, variables_importantes, variable_interes="INC_SMI_REAL"
):
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
    sns.barplot(x="Importancia", y="Modelo", data=importances_df)

    # Títulos y etiquetas
    plt.title(
        f'Importancia de la Variable "{variable_interes}" en las Diferentes Variables Objetivo'
    )
    plt.xlabel("Importancia Promedio")
    plt.ylabel("Modelo")

    # Mostrar el gráfico
    plt.show()


def box_plot_var(df_num, nrows, ncols):
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(16, 10)
    )  # Ajustar filas y columnas

    for ax, (label, values) in zip(axs.flatten(), df_num.items()):
        filtered_values = values.dropna()
        ax.boxplot(filtered_values, vert=False)  # Cambiar orientación
        ax.set_title(label, fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=8)

    # Quitar espacios extra
    plt.tight_layout()
    plt.show()


def plot_real_vs_simulacion(df_real, df_sim, variables, var_plot, n_cols=2):
    n_rows = int(np.ceil(len(var_plot) / n_cols))
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols)
    for i, var in enumerate(var_plot):
        row = i // n_cols
        col = i % n_cols
        ax = axs[row, col]

        ax.plot(df_real["periodo"], df_real[var], color="black", label="Evolución Real")
        ax.plot(
            df_real["periodo"], df_sim[var], color="blue", label="Evolución simulada"
        )
        ax.set_title(f"{var}", fontsize=8)
        ax.set_ylabel(var)
        ax.grid(True)
        ax.legend(fontsize=7, loc="upper right")

    # Ajustar el layout para que no se solapen los títulos y las etiquetas

    plt.suptitle("Evolución real de Madrid vs Simulación")

    # Eliminar subgráficos vacíos
    for i in range(len(variables), n_rows * n_cols):
        fig.delaxes(axs.flatten()[i])

    # Mostrar el gráfico
    plt.tight_layout()
    plt.show()
