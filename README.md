# Description

## Objetivo de la competencia:

El objetivo de esta competencia es construir un modelo para predecir si los ingresos exceden los $50.000 dólares al año a partir de información de empleados en los Estados Unidos. El modelo entrenado debe corresponder a árboles de decisión o bosques aleatorios. Los datos suministrados para el examen son parte del data set "Census Income", pero tienen valores faltantes, los cuales deben ser pre procesados mediante todas las técnicas que usted considere necesarias.

## Data set:

El data set suministrado es una muestra del conocido "Census Income", pero tiene variaciones, el data set para el parcial tiene valores faltantes. El data set compone de 16 atributos, incluyendo la clase ‘income’. A continuación se describen los atributos del conjunto de datos:

- Id: integer
- age: continuous
- workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked
- fnlwgt: continuous (final weight: es el número de personas que el censo cree que representa los valores de la entrada)
- education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
- education-num: continuous.
- marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse
- occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces
- relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried
- race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black
- sex: Female, Male
- capital-gain: continuous
- capital-loss: continuous
- hours-per-week: continuous
- native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands
- Income: low (<=$50.000), high (>$50.000)
Más información de los atributos del data set: https://archive.ics.uci.edu/ml/datasets/Adult

## Data sets entregados:

- NA_Train.csv: 21.113 registros y 16 variables, el data set tiene información de la clase.
- NA_Test.csv: 9.049 registros y 15 variables, el data set no tiene información de la clase, pues es su trabajo predecir esos valores.

Por lo tanto, usted debe entrenar el modelo que mejor precisión brinde a partir de los datos de entrenamiento (NA_Train.csv) para lograr hacer la mejor predicción del nivel de ingreso delos datos que se encuentran en (NA_Test.csv).

Los data sets de entrenamiento (NA_Train.csv) y prueba (NA_Test.csv) tienen valores faltantes, es importante hacer tareas de preprocesamiento para tener un data set de calidad previo al entrenamiento del modelo.

Si quiere ver más en detalle los datos, en la pestaña "Datos", puede explorarlos más detalladamente.

## Problema de Preprocesamiento:

Los datos suministrados tienen valores faltantes que deben ser previamente imputados, junto con técnicas de preprocesamiento, como por ejemplo:

- Técnicas de imputación,
- Técnicas de normalización,
- Técnicas de discretización,
- Entre otras, que usted considere necesarias.

- Problema de Clasificación:
La tarea de predicción consiste en determinar si una persona gana más de $50.000 al año a partir de los 15 atributos. Una vez que usted considere que ha creado un modelo competitivo, su archivo de predicción a Kaggle para ver dónde queda ranqueado su modelo por la precisión frente a los otros compañeros del curso.

- Formato de archivo de envío:
Debe enviar un archivo submission.csv con dos columnas (Id, income) los cuales corresponden al Id de la muestra y la predicción de su modelo, para cada una de las 9.049 muestras del archivo NA_Test.csv.

El archivo a enviar (submission.csv) debe tener exactamente 2 columnas:

- Id: el valor contenido en el csv
- income: contiene sus predicciones: (low, high)

## Evaluation

### Objetivo

Su objetivo es precedir si un empleado tendrá un nivel de salario low (<=$50.000) o high (>$50.000). Para cada uno en el conjunto de prueba, debe predecir un valor de low o high para la variable income.

### Métrica

La evaluación en la presente competencia es la Precisión de clasificación (Classification Accuracy), definida como el número de predicciones correctas dividido por el número de predicciones totales.
Classification Accuracy More Information

### Formato de archivo de envío

Debe enviar un archivo submission.csv con dos columnas (Id, income) los cuales corresponden al Id de la muestra y la predicción de su modelo, para cada una de las 9.049 muestras del archivo NA_Test.csv.

El archivo a enviar (submission.csv) debe tener exactamente 2 columnas:

- Id: el valor contenido en el csv
- income: contiene sus predicciones: (low, high)

El archivo debe contener un encabezado y tener el siguiente formato:

Id,Income
6101,high
25735,low
31995,high
26528,high
16510,low
Etc.
Puede descargar un archivo de envío de ejemplo (sampleSubmission.csv) en la pestaña Datos.

## Exploratory Data Analysis

En el mismo cuardenillo, debe dar respuesta (tipo Markdown) a cada una de las siguientes preguntas, su respuesta debe estar soportada por la respectiva implementación (tipo Code) de la estrategia EDA:

1. Según el dataset, en qué sector se encuentran la mayoría de empleos y a qué porcentaje corresponde.

2. Indique la minoría de personas que tipo de formación tienen en cuanto a la secundaria, títulos
universitarios y licenciatura.

3. ¿Qué relación encuentra entre las personas casadas y las no casadas frente a la situación laboral?.

4. ¿Cómo es la distribución de trabajo según el género y según el rango etario?:
 juventud: 14 - 26 años,  adultez: 27 - 59 años,  vejez: 60 años y más.

5. ¿Cómo es el comportamiento, en relación con la edad, en el tiempo que comienzan a trabajar las
hombres versus las mujeres?