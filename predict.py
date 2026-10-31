import sys
import numpy as np
import tensorflow as tf

"""
Kahve Çekirdeği Tekil Görsel Tahmin Betiği (Inference)
Kullanım: python predict.py <gorsel_yolu>
"""

CLASS_NAMES = ["Dark", "Green", "Light", "Medium"]


def predict_coffee_bean(image_path, model_path="coffee_bean_cnn_model.keras"):
  print(f"Model yükleniyor: {model_path}...")
  model = tf.keras.models.load_model(model_path)

  # Görseli yükleme ve boyutlandırma
  img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
  img_array = tf.keras.utils.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0)  # Batch boyutu ekle: (1, 224, 224, 3)

  # Tahmin
  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])

  predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
  confidence = 100 * np.max(predictions[0])

  print("\n" + "=" * 40)
  print(f"Görsel: {image_path}")
  print(f"Tahmin Edilen Sınıf: {predicted_class}")
  print(f"Güven Oranı: %{confidence:.2f}")
  print("=" * 40)
  return predicted_class, confidence


if __name__ == "__main__":
  if len(sys.argv) > 1:
    img_file = sys.argv[1]
  else:
    img_file = "sample.png"
    print("Kullanım: python predict.py <gorsel_yolu>")
    print("Örnek: python predict.py test_image.png")
