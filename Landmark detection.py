import kagglehub
import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


path = kagglehub.dataset_download("confirm/google-landmark-dataset-v2-micro")
print("Dataset at:", path)


csv_path = None
for file in os.listdir(path):
    if file.lower() == "Train.csv":
        csv_path = os.path.join(path, file)
        break

if csv_path is None:
    raise FileNotFoundError("train.csv not found in dataset folder.")

df = pd.read_csv(csv_path)
print("CSV loaded:", csv_path)
print("Columns:", df.columns)
print(df.head())



csv_path = os.path.join(path, "Train.csv")
df = pd.read_csv(csv_path)
print("CSV Columns:", df.columns)
print(df.head())


images_path = None
for root, dirs, files in os.walk(path):
    if os.path.basename(root).lower() == "train":
        images_path = root
        break

print("Images folder:", images_path)


def load_image(filename):
    img_path = None
    for root, dirs, files in os.walk(images_path):
        if filename in files:
            img_path = os.path.join(root, filename)
            break
    if img_path and os.path.exists(img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        return img
    return None


samples = 500
data, labels = [], []

for _, row in df.sample(samples).iterrows():
    img = load_image(row["filename"])
    if img is not None:
        data.append(img)
        labels.append(row["landmark_id"])

data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

print("Dataset shape:", data.shape, "Labels:", labels.shape)


unique_labels = {label: idx for idx, label in enumerate(np.unique(labels))}
y = np.array([unique_labels[l] for l in labels])
y = to_categorical(y, num_classes=len(unique_labels))


X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)


model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(len(unique_labels), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=32)


loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc*100:.2f}%")


idx = np.random.randint(0, X_test.shape[0])
sample_img = X_test[idx]
plt.imshow(sample_img)
plt.axis("off")

pred = model.predict(np.expand_dims(sample_img, axis=0))
pred_label = list(unique_labels.keys())[np.argmax(pred)]
plt.title(f"Predicted Landmark ID: {pred_label}")
plt.show()
