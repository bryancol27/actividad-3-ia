# ==============================================================================
# ACTIVIDAD ACADÉMICA: APRENDIZAJE SUPERVISADO - ÁRBOLES Y REGLAS DE DECISIÓN
# Caso de Estudio: Predicción de Retrasos en un Sistema de Transporte Masivo
# ==============================================================================
"""
Este script implementa un flujo completo de Ciencia de Datos y Machine Learning:
1. Generación de un dataset sintético realista de transporte masivo (TransMilenio/Metro/SITP).
2. Análisis Exploratorio de Datos (EDA) y preprocesamiento con Scikit-Learn Pipelines.
3. Entrenamiento de un modelo de Clasificación con Árboles de Decisión (DecisionTreeClassifier).
4. Extracción de Reglas de Decisión en formato texto (alineado con la teoría de sistemas basados en reglas).
5. Visualización del árbol de decisión, matriz de confusión e importancia de características.
6. Evaluación cuantitativa (Accuracy, Precision, Recall, F1-Score).
7. Función de inferencia para clasificar nuevos escenarios operativos en tiempo real.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Asegurar codificación utf-8 en terminales de Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Configuración estética general de gráficos
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.autolayout'] = True


# ==============================================================================
# 1. GENERACIÓN DEL DATASET SINTÉTICO
# ==============================================================================
def generar_dataset_transporte(n_samples: int = 1500, random_state: int = 42) -> pd.DataFrame:
    """
    Genera un conjunto de datos sintético representativo de las operaciones diarias
    de un sistema de transporte masivo urbano.

    Parámetros:
    -----------
    n_samples : int, cantidad de registros a simular (por defecto 1500).
    random_state : int, semilla pseudoaleatoria para reproducibilidad.

    Retorna:
    --------
    pd.DataFrame : Conjunto de datos con variables operacionales y nivel de retraso.
    """
    np.random.seed(random_state)

    # 1. Variables Temporales y Ambientales
    horas = np.random.randint(5, 24, size=n_samples) # Horas operativas entre 5:00 y 23:00
    dias = np.random.choice(
        ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'],
        size=n_samples,
        p=[0.16, 0.16, 0.16, 0.16, 0.16, 0.11, 0.09]
    )

    # Determinación lógica de Hora Pico (6:00 - 8:59 y 17:00 - 19:59 de Lunes a Viernes)
    es_hora_pico = [
        1 if (d in ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes']) and ((6 <= h <= 8) or (17 <= h <= 19))
        else 0
        for h, d in zip(horas, dias)
    ]
    es_hora_pico = np.array(es_hora_pico)

    # Clima (Despejado, Lluvia ligera, Lluvia fuerte)
    climas = np.random.choice(
        ['Despejado', 'Lluvia ligera', 'Lluvia fuerte'],
        size=n_samples,
        p=[0.60, 0.28, 0.12]
    )

    # 2. Variables de Infraestructura y Operación
    estaciones = np.random.choice(
        ['Portal Norte', 'Calle 100', 'Portal Américas', 'Ricaurte', 'Portal El Dorado', 'Banderas', 'Héroes'],
        size=n_samples,
        p=[0.18, 0.18, 0.18, 0.16, 0.12, 0.10, 0.08]
    )

    # Afluencia de pasajeros: Dependiente de si es hora pico y estación principal
    base_afluencia = np.where(es_hora_pico == 1, np.random.normal(2500, 400, n_samples), np.random.normal(900, 250, n_samples))
    # Ajuste por estación neurálgica
    factor_estacion = np.isin(estaciones, ['Portal Américas', 'Portal Norte', 'Calle 100']) * 300
    afluencia_pasajeros = np.clip(base_afluencia + factor_estacion, 150, 4000).astype(int)

    # Frecuencia programada de buses (tiempo en minutos entre despachos)
    frecuencia_buses_min = np.where(
        es_hora_pico == 1,
        np.random.choice([3, 4, 5, 6], size=n_samples, p=[0.4, 0.35, 0.15, 0.1]),
        np.random.choice([6, 8, 10, 12, 15], size=n_samples, p=[0.2, 0.3, 0.25, 0.15, 0.1])
    )

    # Incidentes viales (choque, manifestación, varada, semáforo dañado)
    # Mayor probabilidad en lluvia fuerte u hora pico
    prob_incidente = np.where(
        climas == 'Lluvia fuerte', 0.25,
        np.where(es_hora_pico == 1, 0.12, 0.04)
    )
    incidente_vial = (np.random.rand(n_samples) < prob_incidente).astype(int)

    # 3. Modelado Causal de la Variable Objetivo: nivel_retraso
    # Se calcula una puntuación de congestión/retraso basada en principios operativos
    puntaje_retraso = (
        (es_hora_pico * 2.8) +
        (incidente_vial * 4.5) +
        (climas == 'Lluvia fuerte') * 3.2 +
        (climas == 'Lluvia ligera') * 1.1 +
        (afluencia_pasajeros / 1000.0 * 1.0) +
        (frecuencia_buses_min * 0.15) +
        np.random.normal(0, 1.0, size=n_samples) # Ruido estocástico del entorno real
    )

    # Categorización en tres niveles operativos
    # Umbrales calibrados para balance de clases realista
    niveles = []
    for p in puntaje_retraso:
        if p < 4.2:
            niveles.append('Normal')
        elif p < 7.5:
            niveles.append('Retraso_Moderado')
        else:
            niveles.append('Retraso_Critico')

    # Construcción del DataFrame
    df = pd.DataFrame({
        'hora_del_dia': horas,
        'dia_semana': dias,
        'es_hora_pico': es_hora_pico,
        'clima': climas,
        'estacion_origen': estaciones,
        'afluencia_pasajeros': afluencia_pasajeros,
        'frecuencia_buses_min': frecuencia_buses_min,
        'incidente_vial': incidente_vial,
        'nivel_retraso': niveles
    })

    return df


# ==============================================================================
# 2. FUNCIÓN PARA ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ==============================================================================
def realizar_eda(df: pd.DataFrame):
    """
    Imprime estadísticas descriptivas y validaciones de integridad sobre el dataset.
    """
    print("\n" + "="*80)
    print(" 2. ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
    print("="*80)
    print(f"\n[+] Dimensiones del DataFrame: {df.shape[0]} filas x {df.shape[1]} columnas")
    print("\n--- Primeras 5 observaciones ---")
    print(df.head())

    print("\n--- Información general de tipos y nulos ---")
    print(df.info())

    print("\n--- Conteo de Valores Nulos por Columna ---")
    print(df.isnull().sum())

    print("\n--- Resumen Estadístico de Variables Numéricas ---")
    print(df.describe().T)

    print("\n--- Distribución de la Variable Objetivo (nivel_retraso) ---")
    conteo_target = df['nivel_retraso'].value_counts()
    porc_target = df['nivel_retraso'].value_counts(normalize=True) * 100
    resumen_target = pd.DataFrame({'Frecuencia': conteo_target, 'Porcentaje (%)': porc_target.round(2)})
    print(resumen_target)


# ==============================================================================
# 3. PREPROCESAMIENTO Y PIPELINE DE MODELADO
# ==============================================================================
def construir_y_entrenar_pipeline(df: pd.DataFrame):
    """
    Construye el pipeline de preprocesamiento, entrena el Árbol de Decisión y evalúa.
    """
    print("\n" + "="*80)
    print(" 3. PREPROCESAMIENTO Y DIVISIÓN DE DATOS")
    print("="*80)

    # Separación de características predictoras (X) y variable objetivo (y)
    X = df.drop(columns=['nivel_retraso'])
    y = df['nivel_retraso']

    # Identificación de tipos de columnas
    features_numericas = ['hora_del_dia', 'afluencia_pasajeros', 'frecuencia_buses_min']
    features_categoricas = ['dia_semana', 'clima', 'estacion_origen', 'es_hora_pico', 'incidente_vial']

    print(f"Features Numéricas:   {features_numericas}")
    print(f"Features Categóricas: {features_categoricas}")

    # Definición de Transformadores
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), features_numericas),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features_categoricas)
        ],
        remainder='drop'
    )

    # División Estratificada 80% Entrenamiento / 20% Prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"\n[+] Tamaño conjunto de Entrenamiento (Train): {X_train.shape[0]} muestras")
    print(f"[+] Tamaño conjunto de Prueba (Test):          {X_test.shape[0]} muestras")

    # Creación del Pipeline con Árbol de Decisión
    # Se parametriza max_depth=4 y min_samples_leaf=10 para garantizar interpretabilidad
    # y evitar sobreajuste (overfitting).
    tree_clf = DecisionTreeClassifier(
        criterion='gini',
        max_depth=4,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )

    pipeline = Pipeline(steps=[
        ('preprocesamiento', preprocessor),
        ('clasificador', tree_clf)
    ])

    print("\n" + "="*80)
    print(" 4. ENTRENAMIENTO DEL MODELO DE ÁRBOL DE DECISIÓN")
    print("="*80)
    pipeline.fit(X_train, y_train)
    print("\n[OK] Pipeline entrenado con exito.")

    # Extracción de nombres de columnas codificadas por el OneHotEncoder
    ohe_step = pipeline.named_steps['preprocesamiento'].named_transformers_['cat']
    cat_feature_names = ohe_step.get_feature_names_out(features_categoricas)
    all_feature_names = features_numericas + list(cat_feature_names)

    return pipeline, X_train, X_test, y_train, y_test, all_feature_names


# ==============================================================================
# 4. EXTRACCIÓN DE REGLAS DE DECISIÓN Y VISUALIZACIONES
# ==============================================================================
def exportar_reglas_y_visualizaciones(pipeline, X_train, y_train, all_feature_names):
    """
    Genera el reporte de reglas en lenguaje formal/texto y guarda las figuras explicativas.
    """
    print("\n" + "="*80)
    print(" 5. EXTRACCIÓN DE REGLAS DE DECISIÓN (Sistemas Basados en Reglas)")
    print("="*80)

    trained_tree = pipeline.named_steps['clasificador']
    classes = [str(c) for c in trained_tree.classes_]

    # Exportar reglas en formato texto utilizando export_text
    reglas_texto = export_text(
        trained_tree,
        feature_names=all_feature_names,
        spacing=3,
        decimals=2
    )
    print("\n--- Conjunto de Reglas de Decisión Si-Entonces (IF-THEN) Derivadas del Árbol ---")
    print(reglas_texto)

    # 1. Graficar Árbol de Decisión en Alta Resolución
    plt.figure(figsize=(24, 12), dpi=300)
    plot_tree(
        trained_tree,
        feature_names=all_feature_names,
        class_names=classes,
        filled=True,
        rounded=True,
        fontsize=9,
        precision=2
    )
    plt.title("Árbol de Decisión para Predicción de Retrasos en Transporte Masivo", fontsize=16, weight='bold', pad=15)
    arbol_img_path = "arbol_decision_transporte.png"
    plt.savefig(arbol_img_path, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] Gráfico del árbol de decisión guardado en: {arbol_img_path}")

    # 2. Importancia de las Características
    importancias = trained_tree.feature_importances_
    df_importancias = pd.DataFrame({
        'Característica': all_feature_names,
        'Importancia': importancias
    }).sort_values(by='Importancia', ascending=True)

    # Filtrar solo características con importancia > 0
    df_importancias_nonzero = df_importancias[df_importancias['Importancia'] > 0]

    plt.figure(figsize=(10, 6), dpi=150)
    bars = plt.barh(df_importancias_nonzero['Característica'], df_importancias_nonzero['Importancia'], color='#1f77b4', edgecolor='black')
    plt.xlabel('Importancia Relativa (Gini Importance)', fontsize=12, weight='bold')
    plt.ylabel('Variables / Características', fontsize=12, weight='bold')
    plt.title('Importancia de Características en el Modelo de Árbol', fontsize=14, weight='bold', pad=10)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Agregar etiquetas con valores en las barras
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}', ha='left', va='center', fontsize=9)

    plt.xlim(0, max(df_importancias_nonzero['Importancia']) * 1.15)
    importancia_img_path = "importancia_caracteristicas.png"
    plt.savefig(importancia_img_path, bbox_inches='tight')
    plt.close()
    print(f"[OK] Gráfico de importancia de características guardado en: {importancia_img_path}")


# ==============================================================================
# 5. EVALUACIÓN EXHAUSTIVA DEL MODELO
# ==============================================================================
def evaluar_modelo(pipeline, X_test, y_test):
    """
    Calcula métricas de desempeño sobre el conjunto de prueba (Test).
    """
    print("\n" + "="*80)
    print(" 6. EVALUACIÓN DEL MODELO SOBRE EL CONJUNTO DE PRUEBA (TEST)")
    print("="*80)

    # Predicción
    y_pred = pipeline.predict(X_test)
    classes = pipeline.named_steps['clasificador'].classes_

    # Reporte de Clasificación
    print("\n--- Reporte de Clasificación (Precision, Recall, F1-Score, Accuracy) ---")
    report = classification_report(y_test, y_pred, digits=4)
    print(report)

    # Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    plt.figure(figsize=(8, 6), dpi=150)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        annot_kws={'size': 13, 'weight': 'bold'}
    )
    plt.title('Matriz de Confusión - Clasificación de Retrasos', fontsize=14, weight='bold', pad=12)
    plt.xlabel('Etiqueta Predicha por el Modelo', fontsize=12, weight='bold')
    plt.ylabel('Etiqueta Real (Observada)', fontsize=12, weight='bold')
    
    matriz_img_path = "matriz_confusion.png"
    plt.savefig(matriz_img_path, bbox_inches='tight')
    plt.close()
    print(f"[OK] Gráfico de la Matriz de Confusión guardado en: {matriz_img_path}")


# ==============================================================================
# 6. INFERENCIA Y SIMULACIÓN CON NUEVAS OBSERVACIONES
# ==============================================================================
def predecir_nuevos_casos(pipeline):
    """
    Demuestra la aplicación práctica del modelo en producción evaluando
    escenarios operacionales simulados en tiempo real.
    """
    print("\n" + "="*80)
    print(" 7. PRUEBA DE INFERENCIA CON NUEVOS ESCENARIOS EN TIEMPO REAL")
    print("="*80)

    nuevos_escenarios = pd.DataFrame([
        {
            'hora_del_dia': 18,
            'dia_semana': 'Miércoles',
            'es_hora_pico': 1,
            'clima': 'Lluvia fuerte',
            'estacion_origen': 'Portal Américas',
            'afluencia_pasajeros': 3400,
            'frecuencia_buses_min': 4,
            'incidente_vial': 1
        },
        {
            'hora_del_dia': 11,
            'dia_semana': 'Martes',
            'es_hora_pico': 0,
            'clima': 'Despejado',
            'estacion_origen': 'Héroes',
            'afluencia_pasajeros': 650,
            'frecuencia_buses_min': 8,
            'incidente_vial': 0
        },
        {
            'hora_del_dia': 7,
            'dia_semana': 'Lunes',
            'es_hora_pico': 1,
            'clima': 'Lluvia ligera',
            'estacion_origen': 'Calle 100',
            'afluencia_pasajeros': 2800,
            'frecuencia_buses_min': 5,
            'incidente_vial': 0
        }
    ])

    predicciones = pipeline.predict(nuevos_escenarios)
    probabilidades = pipeline.predict_proba(nuevos_escenarios)
    clases = pipeline.named_steps['clasificador'].classes_

    for i, (_, row) in enumerate(nuevos_escenarios.iterrows()):
        print(f"\n[Escenario {i+1}]:")
        print(f"  - Contexto: {row['dia_semana']} a las {row['hora_del_dia']}:00h | {row['clima']} | Estación: {row['estacion_origen']}")
        print(f"  - Operación: Afluencia={row['afluencia_pasajeros']} pas/h | Frecuencia={row['frecuencia_buses_min']} min | Incidente={row['incidente_vial']}")
        print(f"  => RESULTADO PREDICHO: *** {predicciones[i]} ***")
        probs_dict = {cls: f"{prob*100:.1f}%" for cls, prob in zip(clases, probabilidades[i])}
        print(f"  => Probabilidades por clase: {probs_dict}")


# ==============================================================================
# FUNCIÓN PRINCIPAL (MAIN)
# ==============================================================================
def main():
    print("Iniciando Pipeline de Aprendizaje Supervisado para Transporte Masivo...")

    # 1. Generación y Exportación del Dataset
    dataset_path = "dataset_transporte_masivo.csv"
    df = generar_dataset_transporte(n_samples=1500, random_state=42)
    df.to_csv(dataset_path, index=False, encoding='utf-8')
    print(f"\n[OK] Dataset sintético generado con éxito y guardado en: {dataset_path}")

    # 2. Análisis Exploratorio
    realizar_eda(df)

    # 3. Preprocesamiento y Entrenamiento
    pipeline, X_train, X_test, y_train, y_test, all_feature_names = construir_y_entrenar_pipeline(df)

    # 4. Reglas y Visualizaciones
    exportar_reglas_y_visualizaciones(pipeline, X_train, y_train, all_feature_names)

    # 5. Evaluación
    evaluar_modelo(pipeline, X_test, y_test)

    # 6. Nuevas Predicciones
    predecir_nuevos_casos(pipeline)

    print("\n" + "="*80)
    print(" PROCESO COMPLETADO SATISFACTORIAMENTE")
    print("="*80)


if __name__ == "__main__":
    main()
