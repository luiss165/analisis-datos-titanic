import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos limpios
df = pd.read_csv("data/train_limpio.csv")

# Información general
print("Total de pasajeros:", len(df))

print("\nPersonas que sobrevivieron:")
print(df["Survived"].value_counts())

print("\nPorcentaje de supervivencia:")
print(df["Survived"].value_counts(normalize=True) * 100)

# Supervivencia por sexo
print("\nSupervivencia por sexo:")
print(df.groupby("Sex")["Survived"].mean() * 100)

# Supervivencia por clase
print("\nSupervivencia por clase:")
print(df.groupby("Pclass")["Survived"].mean() * 100)

# Gráfica de supervivencia
sns.countplot(data=df, x="Survived")
plt.title("Supervivencia de pasajeros")
plt.xlabel("Sobrevivió")
plt.ylabel("Cantidad de pasajeros")
plt.savefig("supervivencia.png")
plt.show()

# Gráfica de supervivencia por sexo
sns.barplot(data=df, x="Sex", y="Survived")
plt.title("Supervivencia por sexo")
plt.xlabel("Sexo")
plt.ylabel("Promedio de supervivencia")
plt.savefig("supervivencia_sexo.png")
plt.show()

print("\nAnálisis terminado.")