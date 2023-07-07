# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix


# y_test = [0, 0, 1, 1, 0, 1]
# #y_pred = model.predict(x_test)
# y_pred = [1, 0, 1, 0, 0, 1]
# # Convertir las etiquetas a vectores binarios
# #y_pred = np.argmax(y_pred, axis=1)
# #y_test = np.argmax(y_test, axis=1)

# confmatrix = confusion_matrix(y_test, y_pred, normalize='pred')

# # Dandole vida a la matriz
# fig, ax = plt.subplots(figsize=(7,5))
# group_names = ['True Neg','False Pos','False Neg','True Pos']
# group_counts = [f'{value}' for value in confmatrix.flatten()]
# labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names,group_counts)]
# labels = np.asarray(labels).reshape(2,2)
# sns.heatmap(confmatrix, annot=labels, fmt='', cmap='Blues', cbar=False)
# ax.set_xlabel('Predicted labels', fontsize=14)
# ax.set_ylabel('True labels', fontsize=14)
# ax.set_title('Confusion Matrix', fontsize=18)
# plt.show()

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

def bit_matrix(pred, threshold = 0.5):
  r,c=pred.shape
  for i in range(r):
    for j in range(c):
      # print(i,j)
      pred[i][j]=1 if pred[i][j] >= threshold else 0
  return pred


def confusion_matrix_s(predictions:list,reals:list):
  confusion_matrixs = []
  for i in range(len(predictions)):
    if len(reals[i])== len(predictions[i]):
      confusion_matrixs.append(confusion_matrix(y_true=reals[i], y_pred= predictions[i]))
    # else:
    #   predictions.pop(i)
    #   reals.pop(i)
  return confusion_matrixs


def display_confusion_matrix_s(confusion_matrixs:list):
  for confusion_matrix_s in confusion_matrixs:
    confusionMatrixDisplay = ConfusionMatrixDisplay(
        confusion_matrix = confusion_matrix_s
    )

    confusionMatrixDisplay.plot(cmap="Blues")
    plt.show()

def display_confusion_matrix_s_full(confusion_matrixs:list):
  cmm = np.zeros((2,2))

  for i in confusion_matrixs:
    cmm += i
  cmm = [cmm]
  for confusion_matrix_s in cmm:
    confusionMatrixDisplay = ConfusionMatrixDisplay(
        confusion_matrix = confusion_matrix_s
    )

    confusionMatrixDisplay.plot(cmap="Blues")
    plt.show()