from sklearn.metrics import confusion_matrix
import numpy as np
X=3345
Y=3338


def bit_matrix(pred, threshold = 0.5):
  r,c=pred.shape
  for i in range(r):
    for j in range(c):
      pred[i][j]=1 if pred[i][j] >= threshold else 0
  return pred


def get_predictions_mask_for_one_image(model, i:int,pred:dict):
  join_all = np.zeros((X,Y))

  for x in pred[i].keys():
    for y in pred[i][x].keys():
      r,c=pred[i][x][y].shape
      for v in range(r):
        for w in range(c):
          join_all[v+x][w+y] = pred[i][x][y][v][w]
  return join_all


def confusion_matrix_s(predictions:list,reals:list):
  confusion_matrixs = []
  for i in range(len(predictions)):
    if len(reals[i])== len(predictions[i]):
      confusion_matrixs.append(confusion_matrix(y_true=reals[i], y_pred= predictions[i]))
  return confusion_matrixs

# Precision
import matplotlib.pyplot as plt

def calculate_metrics(confusion_matrix):
    # Calculate precision, recall, and F1-score from the confusion matrix
    tp = confusion_matrix[1,1]
    fp = confusion_matrix[0,1]
    tn = confusion_matrix[0,0]
    fn = confusion_matrix[1,0]
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f1_score

def plot_scores(model_names, precision_scores, recall_scores, f1_scores):
    # Crear figura y ejes
    fig, ax = plt.subplots()

    # Crear barras para las medidas de precisión, recobrado y F1-score
    bar_width = 0.25
    bar_positions = np.arange(len(model_names))
    ax.bar(bar_positions, precision_scores, width=bar_width, label='Precisión')
    ax.bar(bar_positions + bar_width, recall_scores, width=bar_width, label='Recobrado')
    ax.bar(bar_positions + 2*bar_width, f1_scores, width=bar_width, label='F1-score')

    # Añadir etiquetas de modelo y leyenda
    ax.set_xticks(bar_positions + bar_width)
    ax.set_xticklabels(model_names)
    ax.legend()

    # Mostrar la figura
    plt.show()

def plot_scores_for_n_cm(cm_all):
  # test = ["rgb", "nir", "3","4","5","6","7","8"]
  names, precisions, recalls, f1_scores = [], [], [], []
  for index, confusion_matrix_s in enumerate(cm_all):
    
    precision, recall, f1_score = calculate_metrics(confusion_matrix_s)
    names.append(index)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1_score)

  # print("Precision:", precision)
  # print("Recall:", recall)
  # print("F1-score:", f1_score)

  plot_scores(names, precisions, recalls, f1_scores)

# Create example confusion matrix
confusion_matrix = np.array([[80, 20], [30, 70]])

precision, recall, f1_score = calculate_metrics(confusion_matrix_s)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)
plot_scores(["rgb"], [precision], [recall], [f1_score])
plot_scores(["rgb"], [precision], [recall], [f1])