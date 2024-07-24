from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt

# Handwritten numbers as 28 x 28 pixels with grayscale values in [0,255]
mnist = fetch_openml("mnist_784")

px_data = np.array(mnist["data"], dtype=int)
labels = mnist["target"]
num_img = len(mnist["data"])

np.savetxt("mnist_pxdata.txt", px_data, delimiter=",", fmt="%d")
np.savetxt("mnist_labels.txt", np.array(labels, dtype=int), delimiter=",", fmt="%d")

np.savetxt("mnist_pxdata_short.txt", px_data[0:2000,:], delimiter=",", fmt="%d")
np.savetxt("mnist_labels_short.txt", np.array(labels[0:2000], dtype=int), delimiter=",", fmt="%d")

if __name__ == "__main__":
    # Plot some random image from dataset
    idx = np.random.randint(0, num_img)

    print(f"Plotting image with index {idx}, label={labels[idx]}")
    img1 = np.array([[px_data[idx][28*r+c] for c in range(0,28)] for r in range(0,28)])

    plt.imshow(img1)
    plt.show()