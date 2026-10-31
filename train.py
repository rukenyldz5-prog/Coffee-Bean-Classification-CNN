"""LBLM368 | Kahve çekirdeği sınıflandırması: yeniden üretilebilir CNN deneyi.

Bu betik GERÇEK Kaggle veri setini indirir, modeli EĞİTİR, test eder.
Raporun eski %96.75 değerini sabit yazmaz. Yeni eğitim metrikleri değişebilir.
"""
import csv
import hashlib
import json
import os
import random
import shutil
from pathlib import Path

import kagglehub
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
from tensorflow.keras import layers, models

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42
OUT = Path('artifacts')
OUT.mkdir(exist_ok=True)


def find_splits(data_dir):
    """Kaggle arşivindeki üç yaygın düzeni destekler.

    1. train / valid / test  (orijinal ayrımı koru)
    2. train / test          (eğitimin bir kısmını valid ayır, testi koru)
    3. Dark / Green / Light / Medium (sınıf bazında üçe ayır)

    Eğitim/valid/test gruplarına aynı dosyanın kopyalanmaması için ayrım kaynak
    listesi üzerinden bir kez yapılır. Ayrımlar sabit seed=42 ile üretilir.
    """
    data_dir=Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f'Kaggle indirmesi bulunamadı: {data_dir}')
    expected=('dark','green','light','medium')
    valid_names=('valid','val','validation')
    image_ext={'.jpg','.jpeg','.png','.bmp','.gif'}

    def class_map(directory):
        if not directory.is_dir():return None
        children={d.name.lower():d for d in directory.iterdir() if d.is_dir()}
        if not all(cls in children for cls in expected):return None
        return {cls:children[cls] for cls in expected}

    def files_of(folder):
        return sorted(f for f in folder.rglob('*') if f.is_file() and f.suffix.lower() in image_ext)

    def safe_prepare(destination):
        # Yalnızca bizim oluşturduğumuz veri ayrımı dizini temizlenir.
        if destination.exists():shutil.rmtree(destination)
        destination.mkdir(parents=True,exist_ok=True)

    def distribute(destination, source, proportion_valid=.125, proportion_test=.25, fixed_test=None):
        safe_prepare(destination)
        for cls in expected:
            images=files_of(source[cls]);random.Random(SEED).shuffle(images)
            if len(images)<8:
                raise ValueError(f'{cls}: yeterli görsel yok ({len(images)})')
            n_test=round(len(images)*proportion_test) if fixed_test is None else 0
            n_val=max(1,round(len(images)*proportion_valid))
            # Tüm listeler ayrık; sınırdan tek kopya her görüntüde
            partitions={'test':images[:n_test],
                        'valid':images[n_test:n_test+n_val],
                        'train':images[n_test+n_val:]}
            if fixed_test is not None:
                partitions['test']=files_of(fixed_test[cls])
            for split,files in partitions.items():
                target=destination/split/cls.capitalize()
                target.mkdir(parents=True,exist_ok=True)
                for i,im in enumerate(files):
                    shutil.copy2(im,target/f'{i:04d}_{im.name}')
        tr,va,te=(destination/key for key in ('train','valid','test'))
        counts={key:sum(len(files_of(part)) for part in class_map(folder).values())
                for key,folder in [('train',tr),('valid',va),('test',te)]}
        print(f'📂 Veri ayrımı hazır: {counts}, toplam={sum(counts.values())}',flush=True)
        return tr,va,te

    # Derinlemesine arama: Kaggle zip dosyalarında bir üst klasör olabiliyor.
    candidates=sorted([data_dir]+[d for d in data_dir.rglob('*') if d.is_dir()],
                      key=lambda d:(len(d.parts),str(d)))
    for location in candidates:
        split_folders={d.name.lower():d for d in location.iterdir() if d.is_dir()}
        tr=split_folders.get('train') or split_folders.get('training')
        te=split_folders.get('test') or split_folders.get('testing')
        va=next((split_folders[x] for x in valid_names if x in split_folders),None)
        if tr is not None and te is not None and class_map(tr) and class_map(te):
            if va is not None and class_map(va):
                print(f'📂 Hazır train/valid/test klasörleri bulundu: {location}',flush=True)
                return tr,va,te
            # Zorunlu: Test verisi eğitim ya da validation'a alınmaz.
            print(f'📂 train/test bulundu, ayrı valid hazırlanıyor: {location}',flush=True)
            return distribute(OUT.parent/'data_split_generated',class_map(tr),
                              proportion_valid=1/6, fixed_test=class_map(te))

    for location in candidates:
        source=class_map(location)
        if source:
            print(f'📂 Dört sınıf klasörü bulundu: {location}',flush=True)
            return distribute(OUT.parent/'data_split_generated',source,
                              proportion_valid=.125,proportion_test=.25)
    discovered=[str(p.relative_to(data_dir)) for p in candidates[:35]]
    raise FileNotFoundError('Kaggle arşivinde ne train/test ne de dört sınıf bulundu. '
                            f'Bulunan ilk klasörler: {discovered}')

def get_dirs(p):
    return find_splits(p)


def count_images(p):
    return sum(f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp') for f in p.rglob('*') if f.is_file())


def plot_learning(hist):
    f, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, a, b, title in ((axes[0], 'accuracy', 'val_accuracy', 'Doğruluk'),
                            (axes[1], 'loss', 'val_loss', 'Kayıp')):
        epochs = range(1, len(hist[a]) + 1)
        ax.plot(epochs, hist[a], marker='o', label='Eğitim')
        ax.plot(epochs, hist[b], marker='o', label='Doğrulama')
        ax.set(xlabel='Epoch', title=title)
        ax.legend();ax.grid(alpha=.25)
    f.tight_layout();f.savefig(OUT/'accuracy_loss_grafikleri.png',dpi=180);plt.close(f)


def sample_examples(rows):
    # Doğru ve yanlış sınıflandırmaları farklı gerçek etiketlerden seçmeye çalışır.
    def pick(kind):
        chosen,seen=[],set()
        for row in rows:
            if row['dogru'] == kind and row['gercek'] not in seen:
                chosen.append(row);seen.add(row['gercek'])
                if len(chosen)==3:return chosen
        for row in rows:
            if row['dogru']==kind and row not in chosen:
                chosen.append(row)
                if len(chosen)==3:break
        return chosen
    return pick(True), pick(False)


def make_prediction_figure(right,wrong):
    panels=[('DOĞRU',r) for r in right]+[('YANLIŞ',r) for r in wrong]
    fig,axes=plt.subplots(2,3,figsize=(12,7.8))
    for ax in axes.flat:ax.axis('off')
    for ax,(status,row) in zip(axes.flat,panels):
        im=tf.keras.utils.load_img(row['dosya'],target_size=IMAGE_SIZE)
        ax.imshow(im)
        ax.set_title(f"{status} | Gerçek: {row['gercek']}\nTahmin: {row['tahmin']}  |  Güven: %{row['guven']*100:.1f}",fontsize=10)
        ax.axis('off')
    if len(panels)<6:
        for ax in axes.flat[len(panels):]:ax.text(.5,.5,'Yeterli yanlış örnek bulunmadı',ha='center',va='center')
    fig.suptitle('Gerçek test görüntülerinden seçilmiş model tahminleri')
    fig.tight_layout(rect=(0,0,1,.95))
    fig.savefig(OUT/'ornek_tahminler.png',dpi=200,bbox_inches='tight');plt.close(fig)


def main():
    tf.keras.utils.set_random_seed(SEED)
    path=kagglehub.dataset_download('gpiosenka/coffee-bean-dataset-resized-224-x-224')
    tr,va,te=get_dirs(path)
    counts={k:count_images(p) for k,p in [('train',tr),('valid',va),('test',te)]}
    train_raw=tf.keras.utils.image_dataset_from_directory(tr,image_size=IMAGE_SIZE,batch_size=BATCH_SIZE,shuffle=True,seed=SEED)
    val_raw=tf.keras.utils.image_dataset_from_directory(va,image_size=IMAGE_SIZE,batch_size=BATCH_SIZE,shuffle=False)
    test_raw=tf.keras.utils.image_dataset_from_directory(te,image_size=IMAGE_SIZE,batch_size=BATCH_SIZE,shuffle=False)
    classes=train_raw.class_names
    assert classes==test_raw.class_names==val_raw.class_names, 'Sınıf sırası aynı değil'
    image_paths=list(test_raw.file_paths)
    autotune=tf.data.AUTOTUNE
    train=train_raw.cache().prefetch(autotune)
    valid=val_raw.cache().prefetch(autotune)
    test=test_raw.cache().prefetch(autotune)
    model=models.Sequential([
        layers.Input((224,224,3)),layers.Rescaling(1.0/255),
        layers.Conv2D(32,3,activation='relu',padding='same'),layers.MaxPooling2D(2),
        layers.Conv2D(64,3,activation='relu',padding='same'),layers.MaxPooling2D(2),
        layers.Conv2D(128,3,activation='relu',padding='same'),layers.MaxPooling2D(2),
        layers.Flatten(),layers.Dense(128,activation='relu'),layers.Dropout(.3),
        layers.Dense(len(classes),activation='softmax')])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001),
                  loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    history=model.fit(train,validation_data=valid,epochs=EPOCHS)
    model.save(OUT/'coffee_bean_cnn_model.keras')
    (OUT/'class_names.json').write_text(json.dumps(classes),encoding='utf-8')
    plot_learning(history.history)
    rows=[]
    offset=0
    y_true=[];y_pred=[]
    for batch_x,batch_y in test:
        probabilities=model.predict(batch_x,verbose=0)
        argmax=np.argmax(probabilities,axis=1)
        for j in range(len(argmax)):
            path=Path(image_paths[offset+j])
            real=int(batch_y[j]);pred=int(argmax[j]);soft=probabilities[j]
            rows.append({'test_indeksi':offset+j,
                         'dosya':str(path),
                         'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                         'gercek':classes[real], 'tahmin':classes[pred],
                         'dogru':real==pred,'guven':float(soft[pred]),
                         **{f'p_{name}':float(soft[i]) for i,name in enumerate(classes)}})
            y_true.append(real);y_pred.append(pred)
        offset += len(argmax)
    assert offset==len(image_paths)
    with open(OUT/'tekil_test_tahminleri.csv','w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        writer.writeheader();writer.writerows(rows)
    right,wrong=sample_examples(rows)
    make_prediction_figure(right,wrong)
    cm=confusion_matrix(y_true,y_pred,labels=range(len(classes)))
    fig,ax=plt.subplots(figsize=(7,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',ax=ax,xticklabels=classes,yticklabels=classes)
    ax.set(xlabel='Tahmin Edilen Sınıf',ylabel='Gerçek Sınıf',title='Karmaşıklık Matrisi')
    fig.tight_layout();fig.savefig(OUT/'confusion_matrix.png',dpi=200);plt.close(fig)
    metrics={'accuracy':float(accuracy_score(y_true,y_pred)),
             'precision_macro':float(precision_score(y_true,y_pred,average='macro',zero_division=0)),
             'recall_macro':float(recall_score(y_true,y_pred,average='macro',zero_division=0)),
             'f1_macro':float(f1_score(y_true,y_pred,average='macro',zero_division=0))}
    output={'dataset_counts':counts,'class_names':classes,'confusion_matrix':cm.tolist(),
            'metrics':metrics,'history':{k:list(map(float,v)) for k,v in history.history.items()},
            'selected_samples':right+wrong,'seed':SEED,'epochs':EPOCHS,
            'experiment_note':'Yeni bir eğitimdir; önceki raporun sayılarıyla zorunlu olarak aynı değildir.'}
    (OUT/'sonuclar.json').write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding='utf-8')
    print(classification_report(y_true,y_pred,target_names=classes,digits=4,zero_division=0))
    print('Çıktılar: artifacts/  |  Test doğruluğu:',metrics['accuracy'])

if __name__=='__main__':
    main()
