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


