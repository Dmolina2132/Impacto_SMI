import pandas as pd
import warnings
from pandas.errors import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def combinar_tablas(
    gasto_basico,
    smi,
    pobreza,
    desigualdad,
    salarios_ocupacion,
    salarios_smis,
    empresas,
    ipc,
    pib_per_capita,
    productividad_hora,
    carencia,
    empleo_hora,
    paro,
    paro_duracion,
    ocupados_jornada,
):
    total_merge = (
        gasto_basico[(gasto_basico["ccaa"] != "Total Nacional")]
        .groupby(["periodo", "ccaa"], as_index=False)
        .sum(numeric_only=True)
        .rename(columns={"Total": "GASTO_BASICO"})
        .merge(smi[["periodo", "smi_14"]], how="outer", on=["periodo"])
    )
    # Riesgo de pobreza
    total_merge = total_merge.merge(
        pobreza[
            (
                pobreza["riesgo_pobreza"]
                == "Tasa de riesgo de pobreza (con alquiler imputado) (renta del año anterior a la entrevista)"
            )
            & (pobreza["ccaa"] != "Total Nacional")
        ][["periodo", "ccaa", "total"]].rename(columns={"total": "RIESGO_POBREZA"}),
        how="outer",
        on=["periodo", "ccaa"],
    )
    # Indice de Gini
    total_merge = total_merge.merge(
        desigualdad[
            (desigualdad["desigualdad"] == "Distribución de la renta S80/S20")
            & (desigualdad["ccaa"] != "Total Nacional")
        ][["periodo", "ccaa", "total"]].rename(columns={"total": "DESIGUALDAD"}),
        how="outer",
        on=["periodo", "ccaa"],
    )
    # Salarios
    total_merge = total_merge.merge(
        salarios_ocupacion[
            (salarios_ocupacion["ccaa"] != "Total Nacional")
            & (salarios_ocupacion["sexo"] == "Ambos sexos")
            & (salarios_ocupacion["ocupacion"] == "Todas las ocupaciones")
        ][["ccaa", "periodo", "salario_año"]],
        how="outer",
        on=["periodo", "ccaa"],
    )
    # Salario SMIs
    salarios_smis_15 = (
        salarios_smis[
            (salarios_smis["ccaa"] != "Total")
            & salarios_smis["smi"].isin(["0-0.5", "0.5-1", "1-1.5"])
        ]
        .rename(columns={"asalariados": "asalariados_15"})
        .groupby(["periodo", "ccaa"], as_index=False)
        .sum(numeric_only=True)
    )
    salarios_smis_total = salarios_smis[
        (salarios_smis["ccaa"] != "Total") & (salarios_smis["smi"] == "Total")
    ].rename(columns={"asalariados": "asalariados_total"})
    salarios_smis_15 = salarios_smis_15.merge(
        salarios_smis_total, how="left", on=["periodo", "ccaa"]
    )
    salarios_smis_15["EMP_1_5"] = (
        salarios_smis_15["asalariados_15"] / salarios_smis_15["asalariados_total"]
    )
    total_merge = total_merge.merge(
        salarios_smis_15[["periodo", "ccaa", "EMP_1_5"]],
        how="outer",
        on=["periodo", "ccaa"],
    )
    # Creamos las tablas para cada índice de empresas correspondiente
    menos_20 = ["De 10 a 19"]
    menos_10 = ["De 1 a 2", "De 3 a 5", "De 6 a 9"]
    menos_50 = ["De 20 a 49"]
    total_empresas = empresas[
        (empresas["ccaa"] != "Total Nacional")
        & (empresas.estrato_asalariados == "Total")
        & (empresas["actividad_principal"] == "Total CNAE")
    ][["periodo", "ccaa", "total_empresas"]]
    empresas_10 = (
        empresas[
            (empresas["ccaa"] != "Total Nacional")
            & empresas.estrato_asalariados.isin(menos_10)
            & (empresas["actividad_principal"] == "Total CNAE")
        ][["periodo", "ccaa", "total_empresas"]]
        .groupby(["periodo", "ccaa"], as_index=False)
        .agg({"total_empresas": "sum"})
        .rename(columns={"total_empresas": "empresas_10"})
    )
    empresas_20 = (
        empresas[
            (empresas["ccaa"] != "Total Nacional")
            & empresas.estrato_asalariados.isin(menos_20)
            & (empresas["actividad_principal"] == "Total CNAE")
        ][["periodo", "ccaa", "total_empresas"]]
        .groupby(["periodo", "ccaa"], as_index=False)
        .agg({"total_empresas": "sum"})
        .rename(columns={"total_empresas": "empresas_20"})
    )
    empresas_50 = (
        empresas[
            (empresas["ccaa"] != "Total Nacional")
            & empresas.estrato_asalariados.isin(menos_50)
            & (empresas["actividad_principal"] == "Total CNAE")
        ][["periodo", "ccaa", "total_empresas"]]
        .groupby(["periodo", "ccaa"], as_index=False)
        .agg({"total_empresas": "sum"})
        .rename(columns={"total_empresas": "empresas_50"})
    )
    total_empresas = total_empresas.merge(
        empresas_10, how="outer", on=["periodo", "ccaa"]
    )
    total_empresas = total_empresas.merge(
        empresas_20, how="outer", on=["periodo", "ccaa"]
    )
    total_empresas = total_empresas.merge(
        empresas_50, how="outer", on=["periodo", "ccaa"]
    )
    total_empresas["EMPRESAS_10"] = (
        total_empresas["empresas_10"] / total_empresas["total_empresas"]
    )
    total_empresas["EMPRESAS_20"] = (
        total_empresas["empresas_20"] / total_empresas["total_empresas"]
    )
    total_empresas["EMPRESAS_50"] = (
        total_empresas["empresas_50"] / total_empresas["total_empresas"]
    )
    total_empresas = total_empresas[
        ["periodo", "ccaa", "EMPRESAS_10", "EMPRESAS_20", "EMPRESAS_50"]
    ]
    total_merge = total_merge.merge(total_empresas, how="outer", on=["periodo", "ccaa"])

    # IPC
    ipc_anual = ipc[
        (ipc["ccaa"] != "Nacional")
        & (ipc["tipo_dato"] == "Índice")
        & (ipc["grupo_indice"] == "Índice general")
        & (ipc["mes"] == 1)
    ][["año", "ccaa", "Total"]].rename(columns={"Total": "IPC", "año": "periodo"})
    total_merge = total_merge.merge(ipc_anual, how="outer", on=["periodo", "ccaa"])

    # Ocupados
    ocupados_sector = pd.read_csv("../../processed_data/trabajo/ocupados_sector.csv")
    ocupados_sector_total = ocupados_sector[
        (ocupados_sector["ccaa"] != "Total Nacional")
        & (ocupados_sector["sexo"] == "Ambos sexos")
        & (ocupados_sector["edad"] == "Total")
        & (ocupados_sector["sector_economico"] == "Total")
    ][["ccaa", "periodo", "Total"]]
    ocupados_construccion = ocupados_sector[
        (ocupados_sector["ccaa"] != "Total Nacional")
        & (ocupados_sector["sexo"] == "Ambos sexos")
        & (ocupados_sector["edad"] == "Total")
        & (ocupados_sector["sector_economico"] == "Construcción")
    ][["ccaa", "periodo", "Total"]].rename(columns={"Total": "ocupados_construccion"})
    ocupados_servicio = ocupados_sector[
        (ocupados_sector["ccaa"] != "Total Nacional")
        & (ocupados_sector["sexo"] == "Ambos sexos")
        & (ocupados_sector["edad"] == "Total")
        & (ocupados_sector["sector_economico"] == "Servicios")
    ][["ccaa", "periodo", "Total"]].rename(columns={"Total": "ocupados_servicio"})
    ocupados_merge = ocupados_sector_total.merge(
        ocupados_servicio, how="left", on=["ccaa", "periodo"]
    )
    ocupados_merge = ocupados_merge.merge(
        ocupados_construccion, how="left", on=["ccaa", "periodo"]
    )
    ocupados_merge["OC_CONSTRUCCION"] = (
        ocupados_merge["ocupados_construccion"] / ocupados_merge["Total"]
    )
    ocupados_merge["OC_SERVICIOS"] = (
        ocupados_merge["ocupados_servicio"] / ocupados_merge["Total"]
    )

    total_merge = total_merge.merge(ocupados_merge, how="outer", on=["periodo", "ccaa"])
    total_merge = total_merge.merge(
        pib_per_capita[
            (pib_per_capita["ccaa"] != "Total Nacional")
            & (pib_per_capita["tipo_dato"] == "Valor")
        ][["ccaa", "periodo", "valor"]].rename(columns={"valor": "PIB_CAPITA"}),
        how="outer",
        on=["periodo", "ccaa"],
    )
    # Productividad por hora
    total_merge = total_merge.merge(
        productividad_hora[(productividad_hora["ccaa"] != "Total Nacional")][
            ["periodo", "ccaa", "total"]
        ].rename(columns={"total": "PROD_HORA"}),
        how="outer",
        on=["periodo", "ccaa"],
    )
    total_merge = total_merge.merge(
        carencia[
            (
                carencia["carencia_material"]
                == "No puede permitirse una comida de carne, pollo o pescado al menos cada dos días"
            )
            & (carencia["ccaa"] != "Total Nacional")
        ][["ccaa", "periodo", "total"]].rename(columns={"total": "CARENCIA"}),
        how="outer",
        on=["periodo", "ccaa"],
    )
    # Horas trabajadas por empleo
    total_merge = total_merge.merge(
        empleo_hora[(empleo_hora["ccaa"] != "Total Nacional")][
            ["periodo", "ccaa", "empleo_hora"]
        ].rename(columns={"empleo_hora": "HORAS_TRABAJO"})
    )

    # Incluimos las variables de paro
    paro_total = paro[
        (paro["ccaa"] != "Total Nacional")
        & (paro["sexo"] == "Ambos sexos")
        & (paro["edad"] == "Total")
    ][["periodo", "ccaa", "tasa_paro_total"]]
    paro_25 = paro[
        (paro["ccaa"] != "Total Nacional")
        & (paro["sexo"] == "Ambos sexos")
        & (paro["edad"] == "Menores de 25 años")
    ][["periodo", "ccaa", "tasa_paro_total"]].rename(
        columns={"tasa_paro_total": "paro_25"}
    )
    paro_25 = paro_25.merge(paro_total, how="left", on=["periodo", "ccaa"]).rename(
        columns={"paro_25": "PARO_25", "tasa_paro_total": "PARO"}
    )
    total_merge = total_merge.merge(paro_25, how="outer", on=["periodo", "ccaa"])

    # Paro de larga duración
    tiempo_1_año = ["De 1 año a menos de 2 años", "2 años o más"]
    total_merge = total_merge.merge(
        paro_duracion[
            (paro_duracion["sexo"] == "Ambos sexos")
            & (paro_duracion["ccaa"] != "Total Nacional")
            & (paro_duracion["tiempo_busqueda"].isin(tiempo_1_año))
        ]
        .groupby(["periodo", "ccaa"], as_index=False)
        .sum(numeric_only=True)
        .rename(columns={"porcentaje_tipo_paro": "PARO_1_AÑO"})
    )
    # Ocupados a jornada parcial
    total_merge = total_merge.merge(
        ocupados_jornada[
            (ocupados_jornada["ccaa"] != "Total Nacional")
            & (ocupados_jornada["sexo"] == "Ambos sexos")
            & (ocupados_jornada["unidad"] == "Porcentaje")
            & (ocupados_jornada["tipo_jornada"] == "Jornada a tiempo parcial")
        ][["periodo", "ccaa", "Total"]].rename(
            columns={"año": "periodo", "Total": "PARCIAL"}
        ),
        how="left",
        on=["periodo", "ccaa"],
    )

    return total_merge


def format_total_merge(total_merge, variables):
    IPC_2015_factor = 100 / total_merge[total_merge["periodo"] == 2015]["IPC"].values[0]
    total_merge["IPC_2015"] = total_merge["IPC"] * IPC_2015_factor
    # Convertimos ahora a nominal
    total_merge["PROD_HORA"] = total_merge["PROD_HORA"] * total_merge["IPC_2015"] / 100
    # Ajustamos ahora por el de 20
    total_merge["PROD_HORA"] = total_merge["PROD_HORA"] / total_merge["IPC"] * 100

    # Ajustamos el pib per capita
    total_merge["PIB_CAPITA"] = total_merge["PIB_CAPITA"] / total_merge["IPC"] * 100

    # Creamos ahora la variable de incremento de SMI
    # Ordenar por región y año
    total_merge = total_merge.sort_values(by=["ccaa", "periodo"])

    # Calcular el salario mínimo real
    total_merge["smi_ajustado"] = total_merge["smi_14"] / total_merge["IPC"] * 100

    # Calcular el salario mínimo del año siguiente
    total_merge["smi_ajustado_next"] = total_merge.groupby("ccaa")[
        "smi_ajustado"
    ].shift(-1)

    # Calcular el incremento porcentual (smi_next / smi - 1)
    total_merge["INC_SMI_REAL"] = (
        total_merge["smi_ajustado_next"] / total_merge["smi_ajustado"] - 1
    )

    # Creamos el SMI_VIDA (para esto podemos dividir directamente los valores nominales, ya que será lo mismo)
    total_merge["SMI_VIDA"] = total_merge["GASTO_BASICO"] / (
        total_merge["smi_14"] * 14
    )  # Multiplicamos porque es en 14 pagas

    # Creamos el SMI_MEDIO
    total_merge["SMI_MEDIO"] = total_merge["smi_14"] * 14 / total_merge["salario_año"]

    df = total_merge[variables]

    # Utilizamos el periodo de 2008 a 2020 (aunque luego lo reduciremos por el hecho de usar incrementos)
    df = df[(df.periodo >= 2008) & (df.periodo <= 2020)]

    # Nos quedamos solo con las comunidades autónomas, excluyendo Ceuta y Melilla
    df = df[~df.ccaa.isin(["Ceuta", "Melilla"])]
    return df


def atrasar_año(
    df,
    var,
    year_col,
    region_col,
    only_diff=["CARENCIA"],
    periodos=1,
    calc_delta=True,
    drop_period_var=True,
):
    # Ordenar los datos por región y año
    df = df.sort_values(by=[region_col, year_col])

    # Crear la columna lag usando groupby para evitar saltos entre regiones
    df[f"{var}_{periodos}"] = df.groupby(region_col)[var].shift(-1 * periodos)

    # Calculamos la variación porcentual
    if calc_delta:
        if var in only_diff:
            df[f"{var}_delta{periodos}"] = df[f"{var}_{periodos}"] - df[f"{var}"]
        else:
            df[f"{var}_delta{periodos}"] = df[f"{var}_{periodos}"] / df[f"{var}"] - 1
    if drop_period_var:
        df.drop(f"{var}_{periodos}", axis=1, inplace=True)

    return df
