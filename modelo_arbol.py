import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Cargar la fuente de datos
try:
    df = pd.read_csv('dataset_transporte.csv')
    print("Datos cargados correctamente.")
except FileNotFoundError:
    print("Error: No se encontró el archivo de datos.")
    exit()

# Convertir variables categóricas (Ruta) a numéricas usando variables dummy
df = pd.get_dummies(df, columns=['Ruta'])

# 2. Separar características (X) y la variable objetivo (y)
X = df.drop('Retraso', axis=1)
y = df['Retraso']

# 3. Dividir los datos en conjunto de entrenamiento y pruebas (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Crear y entrenar el modelo de Árbol de Decisión
modelo = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
modelo.fit(X_train, y_train)

# 5. Realizar predicciones con el conjunto de pruebas
y_pred = modelo.predict(X_test)

# 6. Evaluar los resultados
precision = accuracy_score(y_test, y_pred)
print(f"\nPrecisión del modelo: {precision * 100:.2f}%")
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))