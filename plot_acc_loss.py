import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle

if __name__ == '__main__':

    file_name = sys.argv[1]
    model_name = sys.argv[2]

    names = ['acc', 'val_acc', 'loss', 'val_loss']
    result = list()

    with open(file_name, 'rb') as f:
        metrics = pickle.load(f)

        for name in names:
            result.append(metrics[name])

    result = np.asarray(result, dtype='float32')

    highest = np.max(result[1])
    highest = '{:.2f}%'.format(highest * 100)
    # get training and val error
    result[0] = 1 - result[0]
    result[1] = 1 - result[1]

    # save result
    np.savetxt('{}.csv'.format(file_name), result.T, fmt='%.5f', delimiter=',')

    # plot
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    plt.tight_layout(pad=3, w_pad=2)
    fig.suptitle('{} (val-acc: {})'.format(model_name, highest), fontsize=16, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=14)
    ax1.set_ylabel('Error(%)', fontsize=14)
    ax1.set_ylim(0, 20)
    ax1.plot(result[0]*100, label='Training Error')
    ax1.plot(result[1]*100, label='Validation Error')
    ax1.legend()

    ax2.set_xlabel('Epochs', fontsize=14)
    ax2.set_ylabel('Loss', fontsize=14)
    ax2.set_ylim(0, 1.0)
    ax2.plot(result[2], label='Training Loss')
    ax2.plot(result[3], label='Validation Loss')
    ax2.legend()

    # plt.show()
    plt.savefig('{}.png'.format(file_name))
