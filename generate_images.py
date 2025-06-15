
import os
import numpy as np
import matplotlib.pyplot as plt
import kagglehub

def download_and_generate():
    path = kagglehub.dataset_download("himanshunayal/waferdataset")
    os.makedirs("data/images/pass", exist_ok=True)
    os.makedirs("data/images/fail", exist_ok=True)

    for file in os.listdir(path):
        if file.endswith(".npy"):
            data = np.load(os.path.join(path, file), allow_pickle=True).item()
            label = data['label']
            image = data['waferMap']

            if image is None or image.size == 0:
                continue

            plt.imshow(image, cmap='gray')
            plt.axis('off')
            label_dir = "pass" if label == "Pass" else "fail"
            filename = os.path.join("data/images", label_dir, file.replace(".npy", ".png"))
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            plt.close()

if __name__ == "__main__":
    download_and_generate()
