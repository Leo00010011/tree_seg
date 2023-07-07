import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt


# Crear un DataFrame con los datos
data = pd.DataFrame({'Area': [10, 20, 30, 40, 50],
                     'Rendimiento': [0.55, 0.2, 0.65, 0.55, 0.05],
                     'Tamano': ['Pequeño', 'Pequeño', 'Normmal', 'Normal', 'Normal']})

# Crear el gráfico de violín utilizando Seaborn
sns.violinplot(x='Area', y='Rendimiento', data=data)

# Añadir etiquetas y título
plt.xlabel('Tamaño del área a reconocer')
plt.ylabel('Rendimiento del algoritmo')
plt.title('Rendimiento del algoritmo en función del tamaño del área a reconocer')
plt.show()

violin = sns.catplot(x='Tamano', y='Rendimiento', data=data, kind='violin', color='g')

violin.set_xticklabels(rotation=45)

plt.show()