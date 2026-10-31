"""Eğitilmiş modelle tek görsel tahmini; softmax çıktısı yeniden softmax'tan geçirilmez."""
import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf


def main():
    p=argparse.ArgumentParser()
    p.add_argument('gorsel',help='Test etmek istediğiniz jpg/png görseli')
    p.add_argument('--model',default='artifacts/coffee_bean_cnn_model.keras')
    p.add_argument('--classes',default='artifacts/class_names.json')
    args=p.parse_args()
    classes=json.loads(Path(args.classes).read_text(encoding='utf-8'))
    model=tf.keras.models.load_model(args.model)
    img=tf.keras.utils.load_img(args.gorsel,target_size=(224,224))
    x=np.expand_dims(tf.keras.utils.img_to_array(img),axis=0)
    soft=model.predict(x,verbose=0)[0]
    top=int(np.argmax(soft))
    print('Tahmin edilen sınıf:',classes[top])
    print(f'Softmax güven yüzdesi: %{100*soft[top]:.2f}')
    print('Tüm sınıf olasılıkları:')
    for c,v in zip(classes,soft):print(f'  {c}: %{100*v:.2f}')

if __name__=='__main__':main()
