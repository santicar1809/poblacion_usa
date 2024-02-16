# %% [markdown]
# # Parcial mid

# %%
import pandas as pd
import numpy as np
from numpy import nan
from numpy import isnan
import matplotlib.pyplot as plt
import statistics
import seaborn as sns
import plotly.express as px
from scipy.stats import randint
from sklearn import tree
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import export_graphviz
from scipy.stats import randint
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont
from IPython.display import SVG
import os                               
from pprint import pprint

# %% [markdown]
# # Primera y segunda fase

# %% [markdown]
# **Preprocesamiento**

# %%
dfOrig = pd.read_csv('datasets/NA_fullTrain.csv')
df = dfOrig.copy()
df1=dfOrig.copy()
df.info()

# %%
df.head(29)

# %%
df.columns=['id', 'age', 'workclass', 'fnlwgt', 'education', 'education_num',
       'marital_status', 'occupation', 'relationship', 'race', 'sex',
       'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
       'income']

# %%
num_missing = (df[df.columns] == 0).sum()
num_missing

# %%
sns.heatmap(df.isnull(), cbar=False)

# %% [markdown]
# En las anteriores líneas de código vemos que hay bastantes valores faltantes, devemos calcular cuanto es el porcentaje para actuar de acuerdo a esto.

# %%
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'percent_missing': percent_missing})
missing_value_df.sort_values('percent_missing', inplace=True)
missing_value_df[missing_value_df['percent_missing']>0].sort_values(by='percent_missing',ascending=False)

# %% [markdown]
# Podemos ver que los porcentajes son entre el 4 al 5 %, por lo cual, es posible manejarlo mediante metodos de imputacion.

# %%
df['native_country'].fillna(statistics.mode(df['native_country']),inplace=True)
df['workclass'].fillna(statistics.mode(df['workclass']),inplace=True)
df['education'].fillna(statistics.mode(df['education']),inplace=True)
df['education_num'].fillna(statistics.mode(df['education_num']),inplace=True)
df['marital_status'].fillna(statistics.mode(df['marital_status']),inplace=True)
df['occupation'].fillna(statistics.mode(df['occupation']),inplace=True)
df['relationship'].fillna(statistics.mode(df['relationship']),inplace=True)
df['sex'].fillna(statistics.mode(df['sex']),inplace=True)
df['race'].fillna(statistics.mode(df['race']),inplace=True)

# count the number of NaN values in each column
print("Valores perdidos en normalized_losses: " + str(df.isnull().sum()))


# %%
imputer =KNNImputer(n_neighbors=5, weights="uniform")
# transform the dataset
imputer.fit(df[["age"]])
df["age"]= imputer.transform(df[["age"]])
imputer.fit(df[["fnlwgt"]])
df["fnlwgt"]= imputer.transform(df[["fnlwgt"]])
imputer.fit(df[["capital_gain"]])
df["capital_gain"]= imputer.transform(df[["capital_gain"]])
imputer.fit(df[["capital_loss"]])
df["capital_loss"]= imputer.transform(df[["capital_loss"]])
imputer.fit(df[["hours_per_week"]])
df["hours_per_week"]= imputer.transform(df[["hours_per_week"]])
# count the number of NaN values in each column
print("Valores perdidos en normalized_losses: " + str(df.isnull().sum()))

# %%
sns.heatmap(df.isnull(), cbar=False)

# %%
df.describe()

# %%
sns.heatmap(df1.isnull(), cbar=False)

# %%
df1.describe()

# %% [markdown]
# **DECISION TREE**

# %%
x=df.drop('income', axis=1)
y=df['income']


# %%
x = pd.get_dummies(x, drop_first=True)
x=x.drop(['native_country_Scotland','native_country_Holand-Netherlands'],axis=1)

# %%
x

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# %%
#create One DT
decision_tree = tree.DecisionTreeClassifier(criterion='gini')
#decision_tree = DecisionTreeClassifier(criterion='entropy')
decision_tree.fit(x_train, y_train)
predicitions = decision_tree.predict(x_test)
acc = accuracy_score(y_test, predicitions)
acc

# %%
best_acc = 0

for criterion in "gini", "entropy":
    for max_depth in [2,3,4,5,6]:
        for min_samples_leaf in [5, 10, 20, 30]:
            dtree = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, min_samples_leaf=min_samples_leaf)
            dtree.fit(x_train, y_train)
            predicitions = dtree.predict(x_test)
            acc = accuracy_score(y_test, predicitions)
            if acc > best_acc:
                best_params = f"criterion: {criterion}, max_depth: {max_depth}, min_samples_leaf: {min_samples_leaf}"
                best_acc = acc
print(best_params,best_acc)


# %%
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

fig = plt.figure(figsize=((10,10)))
plot_tree(dtree,
            feature_names = x.columns,
            class_names=['High', 'Low'], 
            impurity=False,
            proportion=True,
            filled=True)
fig.savefig('test.png')

# %% [markdown]
# **RANDOM FOREST**

# %%
clf=RandomForestClassifier(n_estimators=126, random_state=40,max_depth=20,min_samples_split=3)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(x_train,y_train)

# %%
y_pred = clf.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# %% [markdown]
# **Random Search Cross Validation in Scikit_Learn (modelo escogido para el testeo)**

# %%
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(clf.get_params())

# %%
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = randint(1, 11)
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)

# %%
# Use the random grid to search for best hyperparameters
# First create the base model to tune
clf=RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
clf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
clf_random.fit(x_train,y_train)

# %%
base_model = RandomForestClassifier(n_estimators=1000, random_state=42)
base_model.fit(x_train, y_train)
y_pred = base_model.predict(x_test)
base_accuracy=metrics.accuracy_score(y_test, y_pred)
print("Accuracy:",base_accuracy)

# %%
best_random = clf_random.best_estimator_
y_pred = best_random.predict(x_test)
random_accuracy=metrics.accuracy_score(y_test, y_pred)
print("Accuracy:",random_accuracy)

# %%
print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))

# %% [markdown]
# **Grid Search with Cross Validation**

# %%
#Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
clf=RandomForestClassifier()
clf.fit(x_train,y_train)
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = clf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

# %%
# Fit the grid search to the data
grid_search.fit(x_train, y_train)
grid_search.best_params_
{'bootstrap': True,
 'max_depth': 80,
 'max_features': 3,
 'min_samples_leaf': 5,
 'min_samples_split': 12,
 'n_estimators': 100}
best_grid = grid_search.best_estimator_
y_pred = best_grid.predict(x_test)
grid_accuracy=metrics.accuracy_score(y_test, y_pred)
print("Accuracy:",grid_accuracy)

# %%
print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))

# %% [markdown]
# **Testeo**

# %%
dftOrig = pd.read_csv('NA_fullTest.csv')
dft = dftOrig.copy()

# %%
dft.columns=['id', 'age', 'workclass', 'fnlwgt', 'education', 'education_num',
       'marital_status', 'occupation', 'relationship', 'race', 'sex',
       'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']

# %%
dft.head(30)

# %%
num_missing = (dft[dft.columns] == 0).sum()
num_missing

# %%
sns.heatmap(dft.isnull(), cbar=False)

# %%
percent_missing = dft.isnull().sum() * 100 / len(dft)
missing_value_dft = pd.DataFrame({'percent_missing': percent_missing})
missing_value_dft.sort_values('percent_missing', inplace=True)
missing_value_dft[missing_value_dft['percent_missing']>0].sort_values(by='percent_missing',ascending=False)

# %%
dft['native_country'].fillna(statistics.mode(dft['native_country']),inplace=True)
dft['workclass'].fillna(statistics.mode(dft['workclass']),inplace=True)
dft['education'].fillna(statistics.mode(dft['education']),inplace=True)
dft['education_num'].fillna(statistics.mode(dft['education_num']),inplace=True)
dft['marital_status'].fillna(statistics.mode(dft['marital_status']),inplace=True)
dft['occupation'].fillna(statistics.mode(dft['occupation']),inplace=True)
dft['relationship'].fillna(statistics.mode(dft['relationship']),inplace=True)
dft['sex'].fillna(statistics.mode(dft['sex']),inplace=True)
dft['race'].fillna(statistics.mode(dft['race']),inplace=True)

# count the number of NaN values in each column
print("Valores perdidos en normalized_losses: " + str(dft.isnull().sum()))


# %%
imputer =KNNImputer(n_neighbors=5, weights="uniform")
# transform the dataset
imputer.fit(dft[["age"]])
dft["age"]= imputer.transform(dft[["age"]])
imputer.fit(dft[["fnlwgt"]])
dft["fnlwgt"]= imputer.transform(dft[["fnlwgt"]])
imputer.fit(dft[["education_num"]])
dft["education_num"]= imputer.transform(dft[["education_num"]])
imputer.fit(dft[["capital_gain"]])
dft["capital_gain"]= imputer.transform(dft[["capital_gain"]])
imputer.fit(dft[["capital_loss"]])
dft["capital_loss"]= imputer.transform(dft[["capital_loss"]])
imputer.fit(dft[["hours_per_week"]])
dft["hours_per_week"]= imputer.transform(dft[["hours_per_week"]])
# count the number of NaN values in each column
print("Valores perdidos en normalized_losses: " + str(dft.isnull().sum()))

# %%
sns.heatmap(dft.isnull(), cbar=False)

# %%
dft.describe()

# %%
xt = pd.get_dummies(dft, drop_first=True)

# %%
xt

# %%

y_pred = best_random.predict(xt)

# %%
yt=pd.DataFrame(data=y_pred,columns=['income'])

# %%
yt['id']=xt['id']
yt=yt[['id','income']]
yt.head(30)    

# %%
#Random
yt.to_csv('randomized_prediction.csv', index=False)

# %%
#Grid
y_pred1 = best_grid.predict(xt)
yt1=pd.DataFrame(data=y_pred1,columns=['income'])
yt1['id']=xt['id']
yt1=yt1[['id','income']]
yt1.head(30)    


# %%
yt1.to_csv('grid_prediction.csv', index=False)

# %%
y_pred2 = clf.predict(xt)
yt2=pd.DataFrame(data=y_pred2,columns=['income'])
yt2['id']=xt['id']
yt2=yt2[['id','income']]
yt2.head(30)  

# %%
yt2.to_csv('random_forest_prediction', index=False)


# %% [markdown]
# # Tercera fase

# %% [markdown]
# ### 1. Según el dataset, en qué sector se encuentran la mayoría de empleos y a qué porcentaje corresponde.

# %%
P1=df.groupby(['workclass'])['occupation'].count()
P1.sort_values(ascending=False)

# %%
plt.pie(P1, labels=P1.index, shadow=False,
startangle=0, autopct='%1.1f%%',)
plt.axis('equal')
plt.show()


# %%
P2=df.groupby(['workclass','occupation'])['occupation'].count()
P2=pd.DataFrame(P2,columns=None)
P2=P2.rename(columns={'occupation':'count'}).reset_index()
P2

# %%
fig = px.histogram(P2, x="occupation", y="count",
             color='workclass', barmode='group',
             height=400)
fig.show()  


# %% [markdown]
# Según las gráficas, podemos ver en el gráfico de torta que el mayor porcentaje de trabajos lo tiene el sector privado con 75,3%, además en el gráfico de barras se evidencian que cuenta con todos los trabajos y es el sector con la mayoria de personas en cada trabajo a excepción del trabajo de servicios de protección, donde el govierno local tiene la mayoría.. 

# %% [markdown]
# ### 2. Indique la minoría de personas que tipo de formación tienen en cuanto a la secundaria, títulos universitarios y licenciatura.

# %%
Pr=df.groupby('race')['race'].count()
Pr=pd.DataFrame(Pr,columns=None)
Pr=Pr.rename(columns={'race':'count'}).reset_index()
Pr

# %%
fig, ax = plt.subplots(figsize=(20,10), dpi= 80)
ax.vlines(x=Pr.race, ymin=0, ymax=Pr['count'],color='firebrick', alpha=0.7, linewidth=2)
ax.scatter(x=Pr.race, y=Pr['count'], s=75,color='firebrick', alpha=0.7)
ax.set_title('Race', fontdict={'size':22})
ax.set_ylabel('Race')
ax.set_xticks(Pr.race)
ax.set_xticklabels(Pr.race.str.upper(),rotation=65, fontdict={'horizontalalignment': 'right', 'size':12})
ax.set_ylim(0, 30000)
for row in Pr.itertuples():
 ax.text(row.Index, row.count+.5, s=round(row.count, 2),horizontalalignment= 'center', verticalalignment='bottom',fontsize=15)
plt.show()

# %%
P3=df.groupby(['race','education'])['education'].count()
P3=pd.DataFrame(P3,columns=None)
P3=P3.rename(columns={'education':'count'}).reset_index()
P3

# %%
fig = px.histogram(P3, x="education", y="count",
             color='race', barmode='group',
             height=400)
fig.show()

# %%
fig = px.line(P3, x="education", y="count", color='race')
fig.show()

# %% [markdown]
# Teniendo en cuenta la minoría de la población, los indios americanos eskimo tienen más educación en grado de high_school, seguido por alguna universidad y pregrado.
# Los asiaticos tienen más población en pregrado, high_school y alguna universidad.
# La población negra tiene más nivel educativo de hich_school, seguido de alguna universidad y pregrado.
# La población de otras razas, resalta con el nivel de hig_school, seguido por alguna universidad y pregrado.
# La población blanca tiene más nivel educativo en el high_school, alguna universidad, y pregrado.
# Al analizar las minorías podemos concluir que el nivel educativo común es el high school y algunos llegan a la universidad, dejando ver que estas poblaciones no se ven olvidadas y se educan.

# %% [markdown]
# ### 3. ¿Qué relación encuentra entre las personas casadas y las no casadas frente a la situación laboral?.

# %%
condicion=[(df['marital_status']=='Divorced')|(df['marital_status']=='Never_married')|(df['marital_status']=='Separated')|(df['marital_status']=='Widowed'),(df['marital_status']=='Married_civ_spouse')|(df['marital_status']=='Married_spouse_absent')|(df['marital_status']=='Married_AF_spouse')]
values=['No_casado','Casado']
df['Estado_marital']=np.select(condicion,values)
df

# %%
fig = px.histogram(df, x="marital_status",color="marital_status", y="hours_per_week", barmode='group',
             height=400,histfunc="avg")
fig.show()

# %%
fig = px.histogram(df, x="Estado_marital",color="Estado_marital", y="hours_per_week", barmode='group',
             height=400,histfunc="avg")
fig.show()

# %%
P5=df.groupby(['marital_status','income'])['income'].count()
P5=pd.DataFrame(P5,columns=None)
P5=P5.rename(columns={'income':'count'}).reset_index()
fig = px.histogram(P5, x="marital_status",color="income", y="count", barmode='group',
             height=400,histfunc="sum")
fig.show()

# %%
P6=df.groupby(['Estado_marital','income'])['income'].count()
P6=pd.DataFrame(P6,columns=None)
P6=P6.rename(columns={'income':'count'}).reset_index()
fig = px.histogram(P6, x="Estado_marital",color="income", y="count", barmode='group',
             height=400,histfunc="sum")
fig.show()

# %%
fig = px.histogram(df, x="occupation", y="hours_per_week",
             color='marital_status', barmode='group',
             height=400,histfunc="avg")
fig.show()

# %% [markdown]
# En cuanto al análisis laboral, según los gráficos anteriores podemos evidenciar que las personas casadas tienen mayor numero de horas en el trabajo que las casadas, teniendo en cuenta que las personas no casadas incluyen divorciados, viudo y separado, haciendo una suma de horas de las personas casadas, tenemos un total de 43,25 mil, aproximadamente, superando a los no casados, tambien podemos evidenciar que las personas casadas tienen mayores ingresos que los que no están casados, evidenciando la suma de personas que tienen mayores ingresos y están casadas, podemos ver que son 4511, por el contrario las no casadas son 712, en contraste con los que tienen menores ingresos, resaltan los no casados con 9811.

# %% [markdown]
# 
# ### 4. ¿Cómo es la distribución de trabajo según el género y según el rango etario?:
# - juventud: 14 - 26 años, 
# - adultez: 27 - 59 años, 
# - vejez: 60 años y más.

# %%
condicion=[(df['age']>=14)&(df['age']<=26),(df['age']>=27)&(df['age']<=59),(df['age']>=60)]
values=['juventud','adultez','vejez']
df['rango_etario']=np.select(condicion,values)

# %%
P7=df.groupby(['rango_etario','income'])['income'].count()
P7=pd.DataFrame(P7,columns=None)
P7=P7.rename(columns={'income':'count'}).reset_index()
fig = px.histogram(P7, x="rango_etario",color="income", y="count", barmode='group',
             height=400,histfunc="sum")
fig.show()

# %%
fig = px.histogram(df, x="rango_etario",color="sex", y="hours_per_week", barmode='group',
             height=400,histfunc="avg")
fig.show()

# %%
P8=df.groupby(['rango_etario','occupation'])['occupation'].count()
P8=pd.DataFrame(P8,columns=None)
P8=P8.rename(columns={'occupation':'count'}).reset_index()
fig = px.histogram(P8, x="rango_etario",color="occupation", y="count", barmode='group',
             height=400,histfunc="avg")
fig.show()

# %% [markdown]
# En cuanto al rango etario, podemos hacer los siguientes análisis:
# _ Según la primera gráfica podemos ver que la mayoría de personas con ingresos altos se encuentran en la adultez, con un total de 4733 personas, seguidos de los de adultez y juventud, además en cuanto a menores ingresos, también tenemos a la adultez con 10654, seguido de la juventud y la vejez.
# _ En cuanto a el promedio de horas trabajadas según el genero, podemos ver que la adultez de genero masculino, trabaja más horas en promedio con 44,36 horas a la semana, seguido por la vejez y la juventud, por el contrario en el genero femenino los que más horas trabajan se encuentran en la adultez con 39,51 horas a la semana, seguido por la juventud y la adultez.
# _ Por ultimo, los trabajos más destacados en la adultez son las reparaciones, seguido de los gerentes ejecutivos y especialidad profesional, en la juventud se destaca otro servicio, administracion clerical, ventas, por último, la vejez trabaja más en manager ejecutivo, reparador y especialidad profesional.

# %% [markdown]
# 
# ### 5. ¿Cómo es el comportamiento, en relación con la edad, en el tiempo que comienzan a trabajar las hombres versus las mujeres?

# %%
P4=df.groupby(['age','sex'])['sex'].count()
P4=pd.DataFrame(P4,columns=None)
P4=P4.rename(columns={'sex':'count'}).reset_index()

# %%
P9=df.groupby(['age','sex','income'])['income'].count()
P9=pd.DataFrame(P9,columns=None)
P9=P9.rename(columns={'income':'count'}).reset_index()

# %%
fig = px.line(P4, x="age", y="count", color='sex')
fig.show()

# %%
fig = px.histogram(df, x="age",color="sex", y="hours_per_week", barmode='group',
             height=400,histfunc="avg")
fig.show()

# %%
fig = px.histogram(P9, x="age",color="sex", y="count", barmode='group',
             height=400,histfunc="avg")
fig.show()

# %%
fig = px.histogram(P9, x="age",color="income", y="count", barmode='group',
             height=400,histfunc="avg")
fig.show()

# %% [markdown]
# El comportamiento de la edad de las personas es el siguiente:
# _ Según el primer gráfico, podemos ver que tanto como hombres y mujeres a los 38 años es la mayor cantidad de personas que trabajan y el menor valor es a los 17 años, antes de cumlir la mayoría de edad, después de los 38 la cantidad de personas trabajando empieza a decrecer.
# _ Las personas que más promedio de horas tienen en cuanto a los hombres es a los 38 años con 45 horas en promedio y comienza a decrecer el numero de horas a los 60 años, en el caso de las mujeres a los 35 años trabajan más horas con 45 horas a la semana, y decrece a los 56 años.
# _ Por último, la cantidad mayor de personas con mayores ingresos son de los 43 a los 47 años.


