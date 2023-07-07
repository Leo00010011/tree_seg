import numpy as np

X=3345
Y=3338





def get_all_predictions(model,X_train):
  '''Las predicciones se guardan en un diccionario que tiene como primera llave `img_id` o lo que sería la imagen actual, 
  este valor es nuevamente un diccionario que tiene como llave la coordenada `x` del punto `x_0` en el que se comienza 
  este fragmento de la imagen y este igualmente tiene como valor un diccionario cuya llave es la coordenada `y` del punto `y_0` 
  en el que se comienza este fragmento de la imagen, para finalmente obtener la máscara de la predicción de la imagen `img_id` 
  que comienza en `(x_0,y_0)`'''
  l1, l2, l3, l4 = X_train.shape
  # l1=1  ####borrar   #se usan para cortar la ejecucion a menos imagenes o solo partes de ellas
  # # l3=400   ###borrar
  # # l4=400   ####borrar

  predictions:dict={}
  for i in range(l1):
    predictions[i] = {}
    x=0
    index_x=0
    aux_pred_index=[]
    aux_pred=[]
    while(x<l3):
      predictions[i][x]={}
      y=0
      index_y=0
      rx = x
      if l3-160 < x:
        x = l3-160
        index_x = rx-x

      while(y<l4):
        ry = y
        if l4-160 < y:
          y = l4-160
          index_y = ry-y
        aux_pred_index.append((rx,ry,index_x,index_y))
        aux_pred.append(X_train[i][x:x+160][y:y+160])
        print(i,x,y)
        if ry != y:
          break
        y+=160

      if rx != x:
        break
      x+=160
    predictions_i = model.predict(aux_pred)
    for k, elem in enumerate(aux_pred_index):
      rx,ry,index_x,index_y = elem
      predictions[i][rx][ry] = predictions_i[k][index_x:,:][:,index_y:]
    

  return predictions




def get_predictions_mask_for_one_image( i:int,pred:dict):
  '''se le pasan las predicciones y el img_id y con estas se conforma una matriz de np que es la mascara de prediccion de toda la imagen '''
  join_all = np.zeros((X,Y))

  for x in pred[i].keys():
    for y in pred[i][x].keys():
      r,c=pred[i][x][y].shape
      for v in range(r):
        for w in range(c):
          join_all[v+x][w+y] = pred[i][x][y][v][w]
  return join_all
        