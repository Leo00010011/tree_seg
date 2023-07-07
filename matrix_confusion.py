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


# Precision
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def calculate_precision(confusion_matrix):
    true_negatives = confusion_matrix[0, 0]
    true_positives = confusion_matrix[1, 1]
    false_negatives = confusion_matrix[1, 0]
    false_positives = confusion_matrix[0, 1]
    
    true_values = [true_negatives, true_positives]
    predicted_values = [true_negatives + false_negatives, true_positives + false_positives]
    
    precision = precision_score(y_true=true_values, y_pred=predicted_values, zero_division=0)
    
    return precision


def calculate_recall(confusion_matrix):
    true_negatives = confusion_matrix[0, 0]
    true_positives = confusion_matrix[1, 1]
    false_negatives = confusion_matrix[1, 0]
    false_positives = confusion_matrix[0, 1]
    
    true_values = [true_negatives, true_positives]
    predicted_values = [true_negatives + false_negatives, true_positives + false_positives]
    
    recall = recall_score(y_true=true_values, y_pred=predicted_values, zero_division=0)
    
    return recall

def calculate_f1(confusion_matrix):
    precision = calculate_precision(confusion_matrix)
    recall = calculate_recall(confusion_matrix)
    
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1

# Create example confusion matrix
confusion_matrix = np.array([[80, 20], [30, 70]])

# Calculate precision using the method
precision = calculate_precision(confusion_matrix)

# Print the result
print("The precision is:", precision)

# Calculate recall using the method
recall = calculate_recall(confusion_matrix)

# Print the result
print("The recall is:", recall)

# Create example confusion matrix
confusion_matrix = np.array([[80, 20], [30, 70]])

# Calculate F1 score using the method
f1 = calculate_f1(confusion_matrix)

# Print the result
print("The F1 score is:", f1)