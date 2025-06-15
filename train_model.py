
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
from PIL import Image

X, y = [], []

for label in ["pass", "fail"]:
    folder = f"data/images/{label}"
    for file in os.listdir(folder):
        if file.endswith(".png"):
            img = Image.open(os.path.join(folder, file)).resize((32, 32)).convert('L')
            arr = np.array(img).flatten()
            X.append(arr)
            y.append(1 if label == "pass" else 0)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X_train, y_train)

with open("model/model.pkl", "wb") as f:
    pickle.dump(clf, f)
