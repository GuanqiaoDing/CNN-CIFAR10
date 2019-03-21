import numpy as np
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':

    file_name = sys.argv[1]
    names = ['acc', 'val_acc', 'loss', 'val_loss']
    result = list()

    for i in range(4):
        # read csv
        data = np.genfromtxt(
            './train_results/run_.-tag-{}.csv'.format(names[i]),
            missing_values=0, skip_header=True, delimiter=',', dtype=float
        )

        if i == 0 or i == 1:
            data = 1 - data     # get error from acc

        result.append(data[:, -1])

    # save result
    result = np.array(result).T
    np.savetxt('./train_results/{}.csv'.format(file_name), result, fmt='%.5f', delimiter=',')

    # plot
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    plt.tight_layout(pad=3, w_pad=2)
    fig.suptitle('{} Model'.format(file_name), fontsize=16, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=14)
    ax1.set_ylabel('Error(%)', fontsize=14)
    ax1.plot(range(1, 151), result[:, 0]*100, label='Training Error')
    ax1.plot(range(1, 151), result[:, 1]*100, label='Validation Error')
    ax1.legend()

    ax2.set_xlabel('Epochs', fontsize=14)
    ax2.set_ylabel('Loss', fontsize=14)
    ax2.plot(range(1, 151), result[:, 2], label='Training Loss')
    ax2.plot(range(1, 151), result[:, 3], label='Validation Loss')
    ax2.legend()

    # plt.show()
    plt.savefig('./train_results/{}.png'.format(file_name))