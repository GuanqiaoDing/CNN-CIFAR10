import matplotlib.pyplot as plt
import numpy as np

metric_list = [
    'model/vgg_20_1553198546_metrics.csv',
    'model/resnet_20_1553190361_metrics.csv',
    'model/resnet_20_v2_1553191711_metrics.csv',
    'model/resnext_29-a_1553233808_metrics.csv',
    'model/resnext-29-b_1553226679_metrics.csv'
]

name_list = [
    'VGG-20', 'ResNet-20', 'ResNet-20-v2', 'ResNeXt-29-a', 'ResNeXt-29-b'
]

color_list = ['k', 'g', 'y', 'b', 'r']

data = list()

for file in metric_list:
    metric = np.genfromtxt(file, dtype='float32', delimiter=',')
    data.append(metric)
# plot
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
# plt.tight_layout(pad=3, w_pad=2)
fig.suptitle('Compare Models', fontsize=16, fontweight='bold')
ax1.set_xlabel('Epochs', fontsize=14)
ax1.set_ylabel('Error(%)', fontsize=14)
ax1.set_ylim(0, 30)
lines1 = []
for i in range(5):
    ax1.plot(data[i][:, 0]*100, color=color_list[i], linestyle='dashed')
for i in range(5):
    line, = ax1.plot(data[i][:, 1]*100, color=color_list[i])
    lines1.append(line)
ax1.legend(lines1, name_list)

ax2.set_xlabel('Epochs', fontsize=14)
ax2.set_ylabel('Loss', fontsize=14)
ax2.set_ylim(0, 1.5)
lines2 = []
for i in range(5):
    ax2.plot(data[i][:, 2], color=color_list[i], linestyle='dashed')
for i in range(5):
    line, = ax2.plot(data[i][:, 3], color=color_list[i])
    lines2.append(line)
ax2.legend(lines2, name_list)

plt.show()