import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# PASO 1: Carga de datos
try:
    df = pd.read_csv('dataset_transporte.csv')
    print("¡Listo! Los datos se cargaron sin problema.")
except FileNotFoundError:
    print("Error: No se encontró el archivo de datos.")
    exit()

# PASO 2: Preparar las variables 
df = pd.get_dummies(df, columns=['Ruta'])

# PASO 3: Separar datos en (X) e (y)
X = df.drop('Retraso', axis=1)
y = df['Retraso']

# PASO 4: Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PASO 5: entrenar Árbol de Decisión
modelo = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
modelo.fit(X_train, y_train)

# PASO 6: Evaluación de Resultados
y_pred = modelo.predict(X_test)
precision = accuracy_score(y_test, y_pred)
print(f"\nPrecisión del modelo: {precision * 100:.2f}%")
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

