import json
import matplotlib.pyplot as plt


def plot_all(path_load):
    with open(path_load) as f:
        data = json.load(f)
 
    # for i in data.keys():
    #     data[i]=data[i][50:]

    plt.title("Comparation loss-accuracy")
    plt.plot(data['loss'], label='loss')
    plt.plot(data['val_loss'], label='val_loss')
    plt.plot(data['accuracy'], label='accuracy')
    plt.plot(data['val_accuracy'], label='val_accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('loss-accuracy.png')
    plt.show()

    plt.title("Comparation acurracy - binary_accuracy")
    plt.plot(data['accuracy'], label='accuracy')
    plt.plot(data['val_accuracy'], label='val_accuracy')
    plt.plot(data['binary_accuracy'], label='binary_accuracy')
    plt.plot(data['val_binary_accuracy'], label='val_binary_accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('binary_accuracy.png')
    plt.show()

    plt.title("Comparation jacc_loss-jacc_coef")
    plt.plot(data['jacc_loss'], label='jacc_loss')
    plt.plot(data['val_jacc_loss'], label='val_jacc_loss')
    plt.plot(data['jacc_coef'], label='jacc_coef')
    plt.plot(data['val_jacc_coef'], label='val_jacc_coef')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('jacc.png')
    plt.show()


    plt.title("Comparation dice_coef")
    plt.plot(data['dice_coef'], label='dice_coef')
    plt.plot(data['val_dice_coef'], label='val_dice_coef')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('dice_coef.png')
    plt.show()


    plt.title("Comparation sensitivity ")
    plt.plot(data['sensitivity'], label='sensitivity')
    plt.plot(data['val_sensitivity'], label='val_sensitivity')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('sensitivity.png')
    plt.show()

    plt.title("Comparation specificity ")
    plt.plot(data['specificity'], label='specificity')
    plt.plot(data['val_specificity'], label='val_specificity')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('specificity.png')
    plt.show()

# plot_all(path_load = './metrics_json/Metrics First Jacc/loss.json')
# plot_all(path_load = './metrics_json/Metrics New Jacc/loss.json')
# plot_all(path_load = './metrics_json/New Jacc Nir/Metrics/loss.json')
# plot_all(path_load = './metrics_json/Metrics New Nir/loss.json')
# plot_all(path_load = './metrics_json/Metrics New RGB/loss.json')


def plot_all_comb(paths:list,names:list):
    data=[]
    for p in paths:
        with open(p) as f:
            data.append(json.load(f))
    for i in data[3].keys():
        print(i)
        plt.title(f"Comparation {i}")
        for k in range(len(data)):
            plt.plot(data[k][i][50:] if len(data[k][i])>50 else data[k][i][5], label=names[k])
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig(f'{i}.png')
        plt.show()



paths=['./metrics_json/Metrics First Jacc/loss.json',
       './metrics_json/Metrics New Jacc/loss.json',
       './metrics_json/New Jacc Nir/Metrics/loss.json',
       './metrics_json/Metrics New Nir/loss.json',
       './metrics_json/Metrics New RGB/loss.json']
names = ['Metrics First Jacc',
        'Metrics New Jacc',
        'New Jacc Nir',
        'Metrics New Nir',
        'Metrics New RGB']

plot_all_comb(paths,names)

