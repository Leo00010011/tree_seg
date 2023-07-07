import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


y_test = [0, 0, 1, 1, 0, 1]
#y_pred = model.predict(x_test)
y_pred = [1, 0, 1, 0, 0, 1]
# Convertir las etiquetas a vectores binarios
#y_pred = np.argmax(y_pred, axis=1)
#y_test = np.argmax(y_test, axis=1)

confmatrix = confusion_matrix(y_test, y_pred, normalize='pred')

# Dandole vida a la matriz
fig, ax = plt.subplots(figsize=(7,5))
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = [f'{value}' for value in confmatrix.flatten()]
labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names,group_counts)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(confmatrix, annot=labels, fmt='', cmap='Blues', cbar=False)
ax.set_xlabel('Predicted labels', fontsize=14)
ax.set_ylabel('True labels', fontsize=14)
ax.set_title('Confusion Matrix', fontsize=18)
plt.show()