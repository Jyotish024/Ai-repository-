import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

tf.random.set_seed(42)
np.random.seed(42)

IMG_H, IMG_W, IMG_C = 75, 75, 1
NUM_CLASSES = 26
BATCH_SIZE = 64
EPOCHS = 5

def iou_xyxy(boxes1, boxes2):
    boxes1 = tf.convert_to_tensor(boxes1, dtype=tf.float32)
    boxes2 = tf.convert_to_tensor(boxes2, dtype=tf.float32)
    y1 = tf.maximum(boxes1[..., 0], boxes2[..., 0])
    x1 = tf.maximum(boxes1[..., 1], boxes2[..., 1])
    y2 = tf.minimum(boxes1[..., 2], boxes2[..., 2])
    x2 = tf.minimum(boxes1[..., 3], boxes2[..., 3])
    inter_h = tf.maximum(0.0, y2 - y1)
    inter_w = tf.maximum(0.0, x2 - x1)
    inter = inter_h * inter_w
    area1 = tf.maximum(0.0, (boxes1[..., 2] - boxes1[..., 0])) * tf.maximum(0.0, (boxes1[..., 3] - boxes1[..., 1]))
    area2 = tf.maximum(0.0, (boxes2[..., 2] - boxes2[..., 0])) * tf.maximum(0.0, (boxes2[..., 3] - boxes2[..., 1]))
    union = tf.maximum(area1 + area2 - inter, 1e-8)
    return inter / union

def draw_bbox_on_pil(pil_img, ymin, xmin, ymax, xmax, color="red", width=2, normalized=True):
    draw = ImageDraw.Draw(pil_img)
    w, h = pil_img.size
    if normalized:
        left, right = xmin * w, xmax * w
        top, bottom = ymin * h, ymax * h
    else:
        left, right, top, bottom = xmin, xmax, ymin, ymax
    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=width, fill=color)

def show_batch_with_boxes(images, labels_oh, gt_boxes, pred_classes, pred_boxes, title="Predictions vs GT", n=10):
    n = min(n, images.shape[0])
    plt.figure(figsize=(2*n, 4))
    for i in range(n):
        a = (images[i, ..., 0].numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(a, mode="L").convert("RGB")
        y1, x1, y2, x2 = gt_boxes[i].numpy()
        py1, px1, py2, px2 = pred_boxes[i]
        draw_bbox_on_pil(pil_img, y1, x1, y2, x2, color="lime", width=2, normalized=True)
        draw_bbox_on_pil(pil_img, py1, px1, py2, px2, color="red", width=2, normalized=True)
        t = i + 1
        plt.subplot(2, n, t)
        plt.imshow(pil_img)
        plt.axis("off")
        gt_class = int(tf.argmax(labels_oh[i]).numpy())
        pred_class = int(np.argmax(pred_classes[i]))
        plt.title(f"T:{gt_class}  P:{pred_class}")
        plt.subplot(2, n, n + t)
        sample_iou = float(iou_xyxy(pred_boxes[i], gt_boxes[i]).numpy())
        plt.text(0.5, 0.5, f"IoU: {sample_iou:.2f}", ha="center", va="center", fontsize=12)
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def random_bbox_for_digit(image):
    ymin = 0.2 + tf.random.uniform([], -0.05, 0.05)
    xmin = 0.2 + tf.random.uniform([], -0.05, 0.05)
    ymax = 0.8 + tf.random.uniform([], -0.05, 0.05)
    xmax = 0.8 + tf.random.uniform([], -0.05, 0.05)
    bbox = tf.stack([
        tf.clip_by_value(ymin, 0.0, 1.0),
        tf.clip_by_value(xmin, 0.0, 1.0),
        tf.clip_by_value(ymax, 0.0, 1.0),
        tf.clip_by_value(xmax, 0.0, 1.0),
    ])
    return bbox

def preprocess(image, label):
    image = tf.image.resize(image, (IMG_H, IMG_W))
    image = tf.cast(image, tf.float32) / 255.0
    if IMG_C == 1 and image.shape[-1] == 1:
        pass
    elif IMG_C == 1 and image.shape[-1] == 3:
        image = tf.image.rgb_to_grayscale(image)
    elif IMG_C == 3 and image.shape[-1] == 1:
        image = tf.image.grayscale_to_rgb(image)
    label = label - 1
    bbox = random_bbox_for_digit(image)
    label_oh = tf.one_hot(label, NUM_CLASSES)
    return image, (label_oh, bbox)

def get_datasets(batch_size=BATCH_SIZE):
    ds, info = tfds.load("emnist/letters", split=["train", "test"], as_supervised=True, with_info=True)
    train_raw, test_raw = ds
    train_ds = (train_raw
                .shuffle(10_000)
                .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE))
    val_ds = (test_raw
              .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(batch_size)
              .prefetch(tf.data.AUTOTUNE))
    return train_ds, val_ds, info

def build_model(input_shape=(IMG_H, IMG_W, IMG_C), num_classes=NUM_CLASSES):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    class_output = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_output")(x)
    bbox_output = tf.keras.layers.Dense(4, activation="sigmoid", name="bbox_output")(x)
    model = tf.keras.Model(inputs=inputs, outputs=[class_output, bbox_output])
    return model

class MeanIoUBoxes(tf.keras.metrics.Metric):
    def __init__(self, name="mean_iou_boxes", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
    def update_state(self, y_true, y_pred, sample_weight=None):
        ious = iou_xyxy(y_pred, y_true)
        self.total.assign_add(tf.reduce_sum(ious))
        self.count.assign_add(tf.cast(tf.size(ious), tf.float32))
    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)
    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

def train_and_evaluate(epochs=EPOCHS, batch_size=BATCH_SIZE, visualize=True):
    train_ds, val_ds, info = get_datasets(batch_size)
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={"class_output": "categorical_crossentropy", "bbox_output": "mse"},
        loss_weights={"class_output": 1.0, "bbox_output": 1.0},
        metrics={"class_output": ["accuracy"], "bbox_output": [MeanIoUBoxes(), "mse"]},
    )
    print(model.summary())
    for epoch in range(epochs):
        print(f"\n===== EPOCH {epoch+1}/{epochs} =====")
        hist = model.fit(train_ds, epochs=1, validation_data=val_ds, verbose=1)
        results = model.evaluate(val_ds, verbose=0)
        keys = model.metrics_names
        report = dict(zip(keys, results))
        print("Validation metrics:", {k: float(v) for k, v in report.items()})
        if visualize:
            for images, (labels_oh, gt_boxes) in val_ds.take(1):
                pred_classes, pred_boxes = model.predict(images, verbose=0)
                show_batch_with_boxes(
                    images, labels_oh, gt_boxes,
                    pred_classes, pred_boxes,
                    title=f"Epoch {epoch+1}: GT (green) vs Pred (red)", n=10
                )
    save_path = "alphabet_detector"
    model.save(save_path)
    print(f"Model saved to: {save_path}")
    return model

if __name__ == "__main__":
    _ = train_and_evaluate()
