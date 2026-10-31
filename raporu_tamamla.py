"""Yalnızca GERÇEK train.py çıktılarından Word ve varsa PDF raporu günceller.
Önceki deneyin %96.75 metriğini yeni deneyle KARIŞTIRMAZ.
Kullanım: python raporu_tamamla.py --results artifacts/sonuclar.json
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

from docx import Document
from docx.shared import Cm, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT = Path(__file__).resolve().parent


def replace_text(doc, i, txt):
    p = doc.paragraphs[i]
    # Preserve paragraph style and alignment; use the first text run's formatting.
    style = p.runs[0].style if p.runs else None
    p.clear()
    run = p.add_run(txt)
    if style is not None:
        run.style = style
    run.font.name = 'Calibri'
    run.font.size = Pt(9.4 if len(txt) > 270 else 10)
    return p


def replace_image(doc, idx, png, width_cm):
    p = doc.paragraphs[idx]
    p.clear()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(png), width=Cm(width_cm))


def tr(n):
    return f'{n:.2f}'.replace('.', ',')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default=str(ROOT/'artifacts'/'sonuclar.json'))
    parser.add_argument('--template', default=str(ROOT/'rapor_sablonu.docx'))
    parser.add_argument('--output', default=str(ROOT/'artifacts'/'LBLM368_Ruken_YILDIZ_GERCEK_SONUCLAR_RAPORU.docx'))
    args = parser.parse_args()
    results_path = Path(args.results)
    d = json.loads(results_path.read_text(encoding='utf-8'))
    folder = results_path.parent
    for name in ('accuracy_loss_grafikleri.png', 'confusion_matrix.png', 'ornek_tahminler.png', 'tekil_test_tahminleri.csv', 'coffee_bean_cnn_model.keras'):
        if not (folder/name).exists():
            raise FileNotFoundError(f'Yeni eğitimin zorunlu gerçek çıktı dosyası yok: {folder/name}')
    metrics = d['metrics']; cm = d['confusion_matrix']; classes = d['class_names']
    counts = d['dataset_counts']
    if sum(sum(row) for row in cm) != counts['test']:
        raise ValueError('Karmaşıklık matrisi ile test görüntüsü sayısı uyuşmuyor')
    if len(d['selected_samples']) == 0 or len(classes) != 4:
        raise ValueError('Beklenen dört sınıf ya da örnek tahminler bulunamadı')
    doc = Document(args.template)
    if len(doc.paragraphs) != 106:
        raise ValueError('Şablonun paragraf yapısı beklenenden farklı; yanlış şablon olabilir.')

    replace_text(doc, 23, f"• Özellikler: Veri seti dört sınıftan oluşmaktadır: Green (Kavrulmamış), Light (Açık), Medium (Orta) ve Dark (Koyu). Eğitim kümesinde {counts['train']} görsel, doğrulamada {counts['valid']} görsel ve testte {counts['test']} görsel kullanılmıştır. Görüntüler 224×224 RGB formatındadır.")
    replace_text(doc, 68, f"Bu bölümdeki grafikler, karmaşıklık matrisi ve metrikler yeni çalıştırılan CNN eğitiminden elde edilmiştir. {d['epochs']} epoch eğitim yapılmış, önceden ayrılmış {counts['test']} test görüntüsüyle model değerlendirilmiştir (seed={d['seed']}). Eski raporun metrikleri yeni deneyle karıştırılmamıştır.")
    replace_image(doc, 69, folder/'accuracy_loss_grafikleri.png', 15.8)
    h=d['history']; last=-1
    val_loss=h.get('val_loss',[])
    if len(val_loss)>1:
        change='son epoch doğrulama kaybı yükselmiştir' if val_loss[-1]>val_loss[-2] else 'son epoch doğrulama kaybı artmamıştır'
        sentence=f"Eğitim ve doğrulama doğruluğu ile kayıpları Şekil 1'de gösterilmiştir. Eğitim sonunda eğitim doğruluğu %{tr(100*h['accuracy'][last])}, doğrulama doğruluğu %{tr(100*h['val_accuracy'][last])} olmuştur; {change}. Bu gözlem tek başına aşırı öğrenmeyi kanıtlamaz."
    else:
        sentence='Eğitim ve doğrulama performansının epoch boyunca değişimi Şekil 1 üzerinden izlenmektedir.'
    replace_text(doc,71,'Grafik analizi: '+sentence)
    replace_image(doc, 72, folder/'confusion_matrix.png', 8.8)
    replace_text(doc,74,f'Karmaşıklık Matrisi Analizi ({counts["test"]} Test Görüntüsü):')
    for idx, name in zip((75,76,77,78),('Green','Light','Dark','Medium')):
        j=classes.index(name);good=cm[j][j];total=sum(cm[j]);errors=[]
        for k,n in enumerate(cm[j]):
            if k!=j and n:errors.append(f'{n} adet {classes[k]}')
        err='; yanlış tahminler: '+', '.join(errors) if errors else '; yanlış tahmin yok'
        replace_text(doc,idx,f'• {name}: {good}/{total} doğru (%{tr(100*good/total)} başarı){err}.')
    replace_text(doc,79,f"Performans Metrikleri: Accuracy: %{tr(100*metrics['accuracy'])} | Precision (Macro): {metrics['precision_macro']:.4f} | Recall (Macro): {metrics['recall_macro']:.4f} | F1-Score (Macro): {metrics['f1_macro']:.4f}")

    good=[r for r in d['selected_samples'] if r['dogru']]
    bad=[r for r in d['selected_samples'] if not r['dogru']]
    replace_text(doc,82,f"Şekil 3'te yeniden eğitilen modelin gerçek test kümesinden seçilmiş {len(good)} doğru ve {len(bad)} yanlış tahmini gösterilmektedir. Görüntülerin tahminleri ve Softmax skorları modelden hesaplanmış; tekil test kayıtları artifacts/tekil_test_tahminleri.csv dosyasına yazılmıştır.")
    replace_image(doc,83,folder/'ornek_tahminler.png',15.7)
    replace_text(doc,84,'Şekil 3: Yeni eğitilmiş modelin gerçek test görüntülerinden alınmış doğru ve yanlış tahminleri (hesaplanan Softmax skorları).')
    replace_text(doc,85,'Gerçek Tahmin Örneklerinin Analizi:')
    def fmt(r):return f"{r['gercek']}→{r['tahmin']} (%{tr(r['guven']*100)})"
    replace_text(doc,86, 'Testteki doğru örnekler: '+(', '.join(map(fmt,good)) if good else 'Bu grupta örnek bulunamadı')+'. Gösterilen güven değeri tahmin edilen sınıfın Softmax çıktısıdır; doğruluk kanıtı olarak değil, seçilen örneğin model kararı olarak yorumlanmalıdır.')
    replace_text(doc,87, 'Testteki yanlış örnekler: '+(', '.join(map(fmt,bad)) if bad else 'Bu eğitimde seçilebilecek yanlış sınıflandırma bulunamadı')+'. Yanlış tahminlerin düşük güvenli olması her zaman beklenmez; kalibrasyon ayrıca incelenmelidir.')
    replace_text(doc,88, 'Şekil 3 üzerindeki örnekler tek tek test kümesinden seçilmiştir. Aynı örneklerin gerçek etiketleri, tahminleri, sınıf olasılıkları ve SHA256 görsel özetleri CSV kaydındadır.')
    replace_text(doc,89,'Yeniden üretilebilirlik: TensorFlow, veri seti tanımı, sabit tohum, modelin .keras kaydı, sınıf eşleşmeleri ve tüm test örneklerinin CSV çıktısı proje paketindedir. Yeniden eğitimde donanım ve rastgelelik nedeniyle sonuçlar küçük ölçüde değişebilir.')

    correct=sum(cm[i][i] for i in range(len(cm)))
    replace_text(doc,92,f"• Performans Değerlendirmesi: Bağımsız test kümesinde {correct}/{counts['test']} örnek doğru sınıflandırılmıştır (doğruluk %{tr(100*metrics['accuracy'])}, makro F1={metrics['f1_macro']:.4f}). Model sınıflar arasındaki görsel örüntüleri öğrenmiştir; endüstriyel kullanım için farklı ışık, kamera ve gerçek üretim koşullarında ek doğrulama gereklidir.")
    replace_text(doc,93,'• Karşılaşılan Zorluklar: Kavurma derecelerinin birbirine yakın renk ve dokuları sınıflandırma hatalarına neden olabilmektedir. Son epoch eğrileri ve sınıf bazlı karışıklıklar raporda sunulmuştur; güven skorlarının kalibrasyonu ve dış veri testleri ayrıca yapılmalıdır.')

    # Önceki Word şablonundaki 'sonrakiyle birlikte tut' ayarı büyük şekilleri ve
    # bütün analizi tek sayfada tutmaya çalışarak boş sayfa oluşturabiliyor.
    for index in list(range(69, 80)) + list(range(82, 90)):
        doc.paragraphs[index].paragraph_format.keep_with_next = False

    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    doc.save(output)
    print('Güncellenen Word raporu:',output)
    libreoffice=shutil.which('soffice') or shutil.which('libreoffice')
    if libreoffice:
        command=[libreoffice,'-env:UserInstallation=file:///tmp/lo_ruken_generated','--headless','--convert-to','pdf','--outdir',str(output.parent),str(output)]
        try:
            subprocess.run(command,check=True,timeout=120,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            print('Güncellenen PDF raporu:',output.with_suffix('.pdf'))
        except Exception as exc:
            print('PDF dönüştürülemedi, Word dosyası hazır:',exc)
    else:
        print('PDF üretimi için LibreOffice yok; Word dosyası başarıyla oluşturuldu.')

if __name__=='__main__': main()
