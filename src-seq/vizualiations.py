import os
import matplotlib.pyplot as plt




dir = '/Users/dim__gag/git/Genuine_Posed_EmotionRecognition/data_sequences/train_root/fake_angry/H2N2A_age'

images = os.listdir(dir)




# plot all 20 images
nrows = 4
ncols = 5
fig, ax = plt.subplots(nrows, ncols, figsize=(12, 12))
for i in range(nrows):
    for j in range(ncols):
        img = plt.imread(os.path.join(dir, images[i*ncols + j]))
        ax[i][j].imshow(img)
        ax[i][j].axis('off')
plt.show()


