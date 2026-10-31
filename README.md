# Kahve Çekirdeği Kavurma Seviyesi Sınıflandırması — CNN

**İstanbul Arel Üniversitesi · LBLM368 — Derin Öğrenme ve Sınıflandırma Teknikleri**  
**Öğrenci:** Ruken YILDIZ

Bu projede kahve çekirdekleri fotoğrafları dört sınıfta sınıflandırılır: `Dark`, `Green`, `Light`, `Medium`. Model, TensorFlow/Keras ile geliştirilmiş üç evrişim bloğuna sahip özel bir CNN mimarisidir.

## ✅ Gerçek deneyin sonuçları (31 Ekim 2026)

Google Colab (T4 GPU) üzerinde **10 epoch** eğitim yapıldı. Sonuçlar teslim paketindeki `artifacts/sonuclar.json` ve 400 test kaydının bulunduğu `artifacts/tekil_test_tahminleri.csv` dosyalarıyla doğrulanabilir.

| Ölçüt | Sonuç |
|---|---:|
| Toplam görüntü | 1.600 |
| Eğitim / Doğrulama / Test | 1.000 / 200 / 400 |
| Testte doğru / yanlış | **396 / 4** |
| Test Accuracy | **%99,00** |
| Macro Precision | **%99,04** |
| Macro Recall | **%99,00** |
| Macro F1-score | **%99,00** |
| Epoch | 10 |
| Öğrenme oranı | 0.001 |
| Batch size | 32 |
| Rastgele tohum | 42 |

### Sınıf bazında test sonuçları

| Gerçek sınıf | Doğru | Yanlış |
|---|---:|---|
| Dark | 100/100 | 0 |
| Green | 100/100 | 0 |
| Light | 96/100 | 4 görüntü Green tahmini aldı |
| Medium | 100/100 | 0 |

**Önemli:** Daha önceki raporda yer alan %96,75 / 2.000 görüntü bilgileri eski deneyin raporlanmış değerleridir. Güncel, tekrar eğitilmiş modelin raporunda **%99,00 / 1.600 görüntü** vardır. Veri bölmesi farklı olduğundan sonuçlar birebir karşılaştırmalı üstünlük kanıtı değildir.

### Gerçek çıktı görselleri

![Eğitim ve doğrulama grafikleri](accuracy_loss_grafikleri.png)

![Karmaşıklık matrisi](confusion_matrix.png)

![Gerçek test örnekleri](ornek_tahminler.png)

Bu görseller **aynı eğitimden** elde edilen 400 kayıt içinden üretilmiştir. Hatalı sınıflandırılan dört örneğin tamamı Light → Green olmuştur; üçü görselleştirilmiştir. Yanlış tahminlerin Softmax güveni bazı örneklerde yüksektir; bu nedenle güven değeri doğrulukla eşdeğer değildir.

## 📂 Proje yapısı

- `train.py`: Kaggle verisini indirme, veri bölme, modeli eğitme, test etme ve gerçek tahminleri kaydetme.
- `predict.py`: Eğitilmiş modelle tek kahve fotoğrafını sınıflandırma.
- `run_all.py`: Eğitimi, sonuç kontrolünü, rapor üretimini ve teslim ZIP'ini ardışık başlatma.
- `deneyi_dogrula.py`: CSV, Softmax olasılıkları, karmaşıklık matrisi, metrikler ve örnek SHA256 kayıtları arasında tutarlılık denetimi.
- `raporu_tamamla.py`: Grafik ve test metriklerinden Word/PDF raporu üretme.
- `rapor_sablonu.docx`: Raporun yeniden üretiminde kullanılan Word şablonu.
- `artifacts/`: Gerçek deney raporu, sonuçlar ve test CSV'si.
- `LBLM368_KAHVE_DUZELTILMIS_TEK_CALISTIR.ipynb`: Google Colab defteri.

## ▶️ Deneyi tekrar çalıştırma

Google Colab'da `.ipynb` dosyasını yükleyip **Çalışma Zamanı → Tümünü Çalıştır** seçeneğini kullanın. Alternatif olarak Python 3.10–3.12 ile:

```bash
pip install -r requirements.txt
python run_all.py
```

Eğitimden sonra modeli tek fotoğrafla kullanma:

```bash
python predict.py kahve_fotografi.png
```

Sınıf dizilimi `artifacts/class_names.json` dosyasından okunur. Görseller 224×224 RGB boyutuna getirilir; model girişinde `Rescaling(1/255)` katmanı vardır.

### Eğitilmiş model dosyası

**`artifacts/coffee_bean_cnn_model.keras` yaklaşık 155 MB olduğundan GitHub normal dosya yüklemesinin 100 MB sınırını aşıyor.** Model ağırlıkları tam Colab teslim ZIP'inde bulunur; GitHub'da ağırlıklar olmadan kaynak kod + deney çıktıları sunulur. Yeniden eğitim yapılarak `.keras` dosyası oluşturulabilir. Tam model dosyası olmadan `predict.py` doğrudan çalışmaz.

## 🔍 Deney doğrulaması ve sınırlılıklar

`python deneyi_dogrula.py` komutu test kayıtlarını ve metrikleri çapraz kontrol eder. Test görüntülerinin adı ve SHA256 değeri CSV'dedir; kaynak görüntülerin tamamı GitHub'a yüklenmez, Kaggle'dan indirilebilir. Aynı tohum kullanılsa da GPU ve ortam farklarıyla yeni eğitim sonuçları değişebilir.

Modelin %99 başarısı kullanılan test kümesi içindir. Farklı kamera, ışık ve gerçek üretim koşullarında başarı ayrıca sınanmalıdır; sonuçlar sıfır hata veya üretimde doğrudan uygulanabilirlik iddiası değildir.

## 📚 Kaynaklar

1. [Kaggle Coffee Bean Dataset Resized (224×224)](https://www.kaggle.com/datasets/gpiosenka/coffee-bean-dataset-resized-224-x-224)
2. Arboleda, E. R., Fajardo, A. C., & Medina, R. P. (2018). *Classification of coffee bean species using image processing, artificial neural network and K nearest neighbors*. IEEE ICIRD. DOI: [10.1109/ICIRD.2018.8376326](https://doi.org/10.1109/ICIRD.2018.8376326)
3. Chu, B., Yu, K., Zhao, Y., & He, Y. (2018). *Development of Noninvasive Classification Methods for Different Roasting Degrees of Coffee Beans Using Hyperspectral Imaging*. Sensors, 18(4), 1259. DOI: [10.3390/s18041259](https://doi.org/10.3390/s18041259)
4. García Rivas, R. E., Bertarini, P. L. L., & Fernandes, H. (2025). *Automated Coffee Roast Level Classification Using Machine Learning and Deep Learning Models*. Journal of Food Science, 90(9), e70532. DOI: [10.1111/1750-3841.70532](https://doi.org/10.1111/1750-3841.70532)
5. [TensorFlow](https://www.tensorflow.org/), [scikit-learn](https://scikit-learn.org/).

**Yapay zekâ kullanımı:** İlk taslak ve kod düzenlemede Google Gemini/Antigravity; rapor, kaynakların kontrolü ve kodun yeniden üretilebilirliğini geliştirmede ChatGPT kullanılmıştır.
