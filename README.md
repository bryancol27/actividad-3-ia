# Proyecto Académico: Aprendizaje Supervisado con Árboles y Reglas de Decisión
## Aplicación: Predicción y Diagnóstico de Retrasos en un Sistema de Transporte Masivo

Este repositorio contiene la solución completa para la actividad de Aprendizaje Supervisado, orientada a modelar, interpretar y evaluar la operación de un sistema de transporte masivo (ej. TransMilenio, Metro, SITP) utilizando **Árboles de Decisión** y **Extracción de Reglas de Inferencia (IF-THEN)**.

---

## 📁 Estructura del Proyecto

```text
├── main.py                         # Script principal ejecutable (flujo completo de ML)
├── analisis_transporte.ipynb       # Jupyter Notebook interactivo documentado paso a paso
├── requirements.txt                # Dependencias del proyecto
├── dataset_transporte_masivo.csv   # Dataset sintético generado automáticamente (1,500 registros)
├── arbol_decision_transporte.png   # Visualización gráfica en alta resolución del árbol
├── importancia_caracteristicas.png # Gráfico de barras con la importancia de variables (Gini)
├── matriz_confusion.png            # Matriz de confusión para evaluación en conjunto de prueba
└── README.md                       # Documentación técnica y guía de sustentación
```

---

## 🛠️ Requisitos e Instalación

1. **Clonar o abrir el directorio del proyecto**:
   ```bash
   cd d:\proyectos_personales\ibero\actividad4-integencia-artificial
   ```

2. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ejecutar el script principal**:
   ```bash
   python main.py
   ```

---

## 📊 Resumen Metodológico del Flujo de Trabajo

### 1. Generación del Dataset Sintético Realista
El dataset simula 1,500 eventos operativos con distribuciones y correlaciones lógicas:
- **`hora_del_dia`**: Entero entre 5 y 23.
- **`dia_semana`**: Lunes a Domingo.
- **`es_hora_pico`**: Binaria (1 durante franjas pico 6:00-8:59 y 17:00-19:59 de L-V).
- **`clima`**: 'Despejado', 'Lluvia ligera', 'Lluvia fuerte'.
- **`estacion_origen`**: 'Portal Norte', 'Calle 100', 'Portal Américas', 'Ricaurte', 'Portal El Dorado', 'Banderas', 'Héroes'.
- **`afluencia_pasajeros`**: Pasajeros/hora por estación.
- **`frecuencia_buses_min`**: Intervalo programado de despacho en minutos.
- **`incidente_vial`**: Binaria (1 si hay bloqueo, varada o siniestro vial).
- **Variable Objetivo (`nivel_retraso`)**:
  - `Normal`
  - `Retraso_Moderado`
  - `Retraso_Critico`

### 2. Preprocesamiento Robusto con Pipelines de Scikit-Learn
- **Variables Numéricas**: Normalizadas con `StandardScaler`.
- **Variables Categóricas**: Codificadas mediante `OneHotEncoder(handle_unknown='ignore')`.
- **Estratificación**: Partición 80% Train / 20% Test preservando la proporción de clases con `stratify=y` y semilla fija `random_state=42`.

### 3. Modelado y Control de Complejidad
- Algoritmo: `DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_split=20, min_samples_leaf=10)`.
- **Justificación**: Se limita la profundidad a 4 niveles para garantizar la **interpretabilidad humana** (modelo caja blanca) y prevenir el **sobreajuste (overfitting)**.

### 4. Extracción de Reglas de Decisión
A través de `export_text`, se convierten los nodos y umbrales del árbol en reglas deductivas `SI <condición> ENTONCES <clase>`, permitiendo transferir el conocimiento a sistemas expertos basados en reglas.

### 5. Evaluación Cuantitativa
- Reporte con métricas por clase: **Precision**, **Recall**, **F1-Score** y **Accuracy Global**.
- **Matriz de Confusión** visualizada para evaluar falsos positivos y falsos negativos.
- **Importancia de Características (Feature Importance)** calculada a partir de la reducción de impureza de Gini.

### 6. Inferencia en Escenarios Operativos Nuevos
El script somete al pipeline a 3 escenarios de prueba:
1. **Pico + Lluvia fuerte + Incidente vial** $\rightarrow$ Clasifica: `Retraso_Critico` (100% certeza).
2. **Hora valle + Despejado + Sin incidentes** $\rightarrow$ Clasifica: `Normal` (>90% certeza).
3. **Pico matutino + Lluvia ligera + Sin incidentes** $\rightarrow$ Clasifica: `Retraso_Moderado`.

---

## 🎙️ Puntos Clave para la Sustentación en Video Grupal

1. **¿Por qué Árboles de Decisión?**  
   Son modelos transparentes que permiten explicar la causalidad del retraso a los operadores del sistema de transporte, a diferencia de modelos tipo caja negra.
2. **Importancia de las Variables:**  
   Las variables determinantes son la presencia de incidentes viales (`incidente_vial`), la hora pico (`es_hora_pico`) y el clima adverso (`clima_Lluvia fuerte`).
3. **Mapeo a Reglas de Negocio:**  
   El árbol genera reglas auditables que el centro de control puede programar directamente como alertas tempranas.
