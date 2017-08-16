import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from CNN import conv_model
from sklearn.utils import resample



model = conv_model()
model.load_weights("./output/weights.hdf5")

SignNames = pd.read_csv('./input/signnames.csv')
testing_file = './input/test.p'
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
X_test, y_test = test['features'], test['labels']

X_sample, y_sample = resample(X_test, y_test, n_samples=12)
gs1 = gridspec.GridSpec(4, 3)
gs1.update(wspace=0.005, hspace=0.01) # set the spacing between axes.
plt.figure(figsize=(12,12))
for i in range(12):
    ax1 = plt.subplot(gs1[i])
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_aspect('equal')
    plt.subplot(4,3,i+1)
    plt.imshow(X_sample[i])
    plt.title(SignNames[SignNames['ClassId'] == y_sample[i]]['SignName'].values[0])
    plt.axis('off')

y_prob = model.predict(X_sample)
for i in range(12):
    plt.figure(figsize = (5,1.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2,3])
    plt.subplot(gs[0])
    plt.imshow(X_sample[i])
    plt.axis('off')
    plt.subplot(gs[1])
    top5_ind = np.argpartition(y_prob[i], -5)[-5:]
    plt.barh(6-np.arange(5),y_prob[i][top5_ind], align='center')
    for i_label in range(5):
        plt.text(y_prob[i][top5_ind][i_label]+.02,6-i_label-.25,
            SignNames[SignNames['ClassId'] == top5_ind[i_label]]['SignName'].values[0])
    plt.axis('off')
    plt.text(0,6.95,SignNames[SignNames['ClassId'] == y_sample[i]]['SignName'].values[0])
    plt.show()

