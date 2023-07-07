import json
import matplotlib.pyplot as plt



with open('loss.json') as f:
    data = json.load(f)

# for key in data.keys():
#     print(key)


#     fig, ax = plt.subplots()
#     ax.plot(data[key])
#     ax.set_title(key)
#     ax.set_xlabel("value")
#     ax.set_ylabel('epoch')

#     plt.show()



plt.title("Comparation loss-accuracy")
plt.plot(data['loss'], label='loss')
plt.plot(data['val_loss'], label='val_loss')
plt.plot(data['accuracy'], label='accuracy')
plt.plot(data['val_accuracy'], label='val_accuracy')
plt.xlabel('epoch')
plt.legend()

plt.show()


# plt.title("Comparation accuracy")
# plt.plot(data['accuracy'], label='accuracy')
# plt.plot(data['val_accuracy'], label='val_accuracy')
# plt.xlabel('epoch')
# plt.legend()

# plt.show()

plt.title("Comparation acurracy - binary_accuracy")
plt.plot(data['accuracy'], label='accuracy')
plt.plot(data['val_accuracy'], label='val_accuracy')
plt.plot(data['binary_accuracy'], label='binary_accuracy')
plt.plot(data['val_binary_accuracy'], label='val_binary_accuracy')
plt.xlabel('epoch')
plt.legend()

plt.show()

plt.title("Comparation jacc_loss-jacc_coef")
plt.plot(data['jacc_loss'], label='jacc_loss')
plt.plot(data['val_jacc_loss'], label='val_jacc_loss')
plt.plot(data['jacc_coef'], label='jacc_coef')
plt.plot(data['val_jacc_coef'], label='val_jacc_coef')
plt.xlabel('epoch')
plt.legend()

plt.show()

# plt.title("Comparation jacc_coef")
# plt.plot(data['jacc_coef'], label='jacc_coef')
# plt.plot(data['val_jacc_coef'], label='val_jacc_coef')
# plt.xlabel('epoch')
# plt.legend()

# plt.show()

plt.title("Comparation dice_coef")
plt.plot(data['dice_coef'], label='dice_coef')
plt.plot(data['val_dice_coef'], label='val_dice_coef')
plt.xlabel('epoch')
plt.legend()

plt.show()


plt.title("Comparation sensitivity ")
plt.plot(data['sensitivity'], label='sensitivity')
plt.plot(data['val_sensitivity'], label='val_sensitivity')
plt.xlabel('epoch')
plt.legend()

plt.show()

plt.title("Comparation specificity ")
plt.plot(data['specificity'], label='specificity')
plt.plot(data['val_specificity'], label='val_specificity')
plt.xlabel('epoch')
plt.legend()

plt.show()