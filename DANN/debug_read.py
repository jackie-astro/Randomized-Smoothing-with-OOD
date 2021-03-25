import scipy.io as scio
from matplotlib import pyplot as plt
import numpy as np


mat_path = 'dataset/USPS/usps_resampled.mat'

data = scio.loadmat(mat_path)

print(data.keys())

train_data = data['train_patterns']
train_label = data['train_labels']
data = train_data[:,0]
data = np.resize(data,(16,16))
print(data.shape)
plt.imshow(data)
plt.show()


print(train_data.shape)
print(train_label.shape)