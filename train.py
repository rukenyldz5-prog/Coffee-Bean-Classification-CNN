import os
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models

"""
LBLM368 - Derin Öğrenme ve Sınıflandırma Teknikleri Dönem Projesi
Proje: Evrişimli Sinir Ağları (CNN) ile Kahve Çekirdeği Kavurma Seviyelerinin Çok Sınıflı Sınıflandırılması
Öğrenci: Ruken YILDIZ (230309047)
"""


def main():
  # 1. VERİ SETİNİ İNDİRME
  print("1/6: Veri seti indiriliyor...")
  dataset_path = kagglehub.dataset_download(
      "gpiosenka/coffee-bean-dataset-resized-224-x-224"
  )
  print(f"Veri seti konumu: {dataset_path}")

  def find_split_dirs(base):
    for root, dirs, files in os.walk(base):
      if "train" in dirs:
        tr = os.path.join(root, "train")
        val = os.path.join(root, "valid" if "valid" in dirs else "test")
        te = os.path.join(root, "test" if "test" in dirs else "valid")
        return tr, val, te
    return None, None, None

  train_dir, val_dir, test_dir = find_split_dirs(dataset_path)

  IMG_SIZE = (224, 224)
  BATCH_SIZE = 32

  # 2. VERİLERİ YÜKLEME VE PREFETCH OPTİMİZASYONU
  print("2/6: Görseller yükleniyor ve tf.data optimizasyonu uygulanıyor...")
  train_ds = tf.keras.utils.image_dataset_from_directory(
      train_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True
  )
  val_ds = tf.keras.utils.image_dataset_from_directory(
      val_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
  )
  test_ds = tf.keras.utils.image_dataset_from_directory(
      test_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
  )

  class_names = train_ds.class_names
  print("Sınıflar:", class_names)

  # Performans optimizasyonu (Cache ve Prefetch)
  AUTOTUNE = tf.data.AUTOTUNE
  train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
  val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
  test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

  # 3. CNN MODEL MİMARİSİ
  print("3/6: CNN Modeli kuruluyor...")
  model = models.Sequential([
      layers.Input(shape=(224, 224, 3)),
      layers.Rescaling(1.0 / 255),
      layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
      layers.MaxPooling2D((2, 2)),
      layers.Flatten(),
      layers.Dense(128, activation="relu"),
      layers.Dropout(0.3),
      layers.Dense(len(class_names), activation="softmax"),
  ])

  model.compile(
      optimizer="adam",
      loss="sparse_categorical_crossentropy",
      metrics=["accuracy"],
  )
  model.summary()

  # 4. EĞİTİM (Aşırı öğrenmeyi izleme)
  print("4/6: Model eğitiliyor (10 epoch)...")
  history = model.fit(train_ds, validation_data=val_ds, epochs=10)

  # Eğitilen Modeli Kaydetme
  model.save("coffee_bean_cnn_model.keras")
  print("Model başarıyla 'coffee_bean_cnn_model.keras' olarak kaydedildi.")

  # 5. METRİKLER VE GRAFİKLER
  print("5/6: Sonuçlar değerlendiriliyor ve grafikler çiziliyor...")
  epochs_range = range(1, 11)

  plt.figure(figsize=(12, 5))
  plt.subplot(1, 2, 1)
  plt.plot(
      epochs_range,
      history.history["accuracy"],
      label="Eğitim Doğruluğu",
      marker="o",
  )
  plt.plot(
      epochs_range,
      history.history["val_accuracy"],
      label="Doğrulama Doğruluğu",
      marker="o",
  )
  plt.title("Eğitim ve Doğrulama Doğruluğu")
  plt.xlabel("Epoch")
  plt.ylabel("Doğruluk (Accuracy)")
  plt.legend()
  plt.grid(True)

  plt.subplot(1, 2, 2)
  plt.plot(
      epochs_range, history.history["loss"], label="Eğitim Kaybı", marker="o"
  )
  plt.plot(
      epochs_range,
      history.history["val_loss"],
      label="Doğrulama Kaybı",
      marker="o",
  )
  plt.title("Eğitim ve Doğrulama Kaybı")
  plt.xlabel("Epoch")
  plt.ylabel("Kayıp (Loss)")
  plt.legend()
  plt.grid(True)
  plt.tight_layout()
  plt.savefig("accuracy_loss_grafikleri.png", dpi=300)
  plt.close()

  # Test Değerlendirmesi
  y_true, y_pred = [], []
  for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy())

  y_true = np.array(y_true)
  y_pred = np.array(y_pred)

  print("\n--- PERFORMANS METRİKLERİ ---")
  print(
      classification_report(y_true, y_pred, target_names=class_names, digits=4)
  )

  # Confusion Matrix
  cm = confusion_matrix(y_true, y_pred)
  plt.figure(figsize=(7, 6))
  sns.heatmap(
      cm,
      annot=True,
      fmt="d",
      cmap="Blues",
      xticklabels=class_names,
      yticklabels=class_names,
  )
  plt.title("Karmaşıklık Matrisi (Confusion Matrix)")
  plt.xlabel("Tahmin Edilen Sınıf")
  plt.ylabel("Gerçek Sınıf")
  plt.tight_layout()
  plt.savefig("confusion_matrix.png", dpi=300)
  plt.close()

  print("6/6: Tüm eğitim, değerlendirme ve görselleştirme tamamlandı!")


if __name__ == "__main__":
  main()
