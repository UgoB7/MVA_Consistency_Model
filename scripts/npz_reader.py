import numpy as np
import matplotlib.pyplot as plt

# Charger le fichier NPZ
data = np.load("/tmp/openai-2025-03-17-16-41-17-771428/inpainted_7x256x256.npz")
print("Clés dans le fichier NPZ :", data.files)
images = data[data.files[0]]
print("Shape des images :", images.shape)  # Par exemple (5, 256, 256, 3)

# Afficher et sauvegarder chaque image
for i in range(images.shape[0]):
    plt.figure()
    plt.imshow(images[i])
    plt.title(f"Image {i}")
    # Sauvegarder l'image dans un fichier (optionnel)
    plt.savefig(f"image_{i}.png")
plt.show()
