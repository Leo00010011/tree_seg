# cell for traning
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import os
from model import get_unet
import json
import matplotlib.pyplot as plt
import keras

temp = {}
figPath = '/content/drive/MyDrive/Metrics/loss'
jsonPath = '/content/drive/MyDrive/Metrics/loss.json'
startAt = 0
class TrainingMonitor(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
      # initialize the history dictionary
      temp = {}
      if jsonPath is not None:
        if os.path.exists(jsonPath):
          temp = json.loads(open(jsonPath).read())
          # check to see if a starting epoch was supplied
          if startAt > 0:
              # loop over the entries in the history log and
              # trim any entries that are past the starting
              # epoch
              for k in temp.keys():
                temp[k] = temp[k][:startAt]

    def on_epoch_end(self, epoch, logs={}):
      # loop over the logs and update the loss, accuracy, etc.
      # for the entire training process
      for (k, v) in logs.items():
        l = temp.get(k, [])
        l.append(v)
        temp[k] = l
      # check to see if the training history should be serialized
      # to file
      if jsonPath is not None:
        f = open(jsonPath, "w")
        f.write(json.dumps(temp))
        f.close()
      # ensure at least two epochs have passed before plotting
      # (epoch starts at zero)
      if len(temp["loss"]) > 1:
        # plot the training loss and accuracy
        N = np.arange(1, len(temp["loss"]) + 1)
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, temp["loss"], label="train_loss")
        plt.plot(N, temp["val_loss"], label="val_loss")
        plt.plot(N, temp["accuracy"], label="train_accuracy")
        plt.plot(N, temp["val_accuracy"], label="val_accuracy")
        plt.title("Training Loss and Accuracy [Epoch {}]".format(len(temp["loss"])))
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        # save the figure
        plt.savefig(figPath)

        plt.close()


def train_model(X_train,y_train,X_val,y_val,loss_func,best_weight_path,last_weight_path,step_per_epoch,train_batch_size,test_batch_size)
    img_rows = 160
    img_cols = 160
    _,_,_,ch = X_train.shape
    model = get_unet(ch)
    model.compile(optimizer = Adam(learning_rate=1e-5), loss = loss_func,
                      metrics = ['accuracy', acc, jacc_loss, dice_coef, jacc_coef, sensitivity, specificity])

    callbacks = [
        TrainingMonitor(),
        ModelCheckpoint(best_weight_path, monitor="val_loss", mode="min", save_best_only=True, verbose=1),
        ModelCheckpoint(last_weight_path)
    ]


    H = model.fit(DataGenerator(X_train, y_train, train_batch_size,img_rows,img_cols,ch,steps_per_epoch= step_per_epoch, horizontal_flip=True, vertical_flip=True, swap_axis=True),
                        epochs=100,
                        verbose=1,
                        steps_per_epoch= step_per_epoch,
                        callbacks=callbacks,
                        workers=8,
                        validation_data = DataGenerator(X_val, y_val, test_batch_size,img_rows,img_cols,ch,steps_per_epoch= step_per_epoch, horizontal_flip=True, vertical_flip=True, swap_axis=True)
                        )