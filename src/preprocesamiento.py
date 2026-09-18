import pandas as pd

# Cargar los datos
df = pd.read_csv("data/train.csv")

# Mostrar datos faltantes antes de la limpieza
print("Datos faltantes antes de la limpieza:")
print(df.isnull().sum())

# Rellenar Age con la mediana
df["Age"] = df["Age"].fillna(df["Age"].median())

# Rellenar Embarked con la moda
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Eliminar columnas con poca utilidad para el análisis
df = df.drop(columns=["Cabin", "PassengerId"])

# Guardar el dataset limpio
df.to_csv("data/train_limpio.csv", index=False)

# Mostrar datos faltantes después de la limpieza
print("\nDatos faltantes después de la limpieza:")
print(df.isnull().sum())

print("\nPreprocesamiento terminado.")