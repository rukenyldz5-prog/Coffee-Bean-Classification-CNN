"""Bağımsız ve stdlib ile çalışan deney kanıt tutarlılığı denetimi.
Sınıf doğruluğu, yanlış/dogru bayrakları ve JSON-CSV-görsel ilişkisini sınar.
Bu test, tek başına model eğitiminin yerine GEÇMEZ.
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def check(output=None):
    out = Path(output) if output else ROOT / 'artifacts'
    d = json.loads((out / 'sonuclar.json').read_text(encoding='utf-8'))
    classes = list(d['class_names'])
    assert len(classes) == 4 and len(set(classes)) == 4, 'Dört farklı sınıf gerekli'
    with (out / 'tekil_test_tahminleri.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    cm = [[0] * len(classes) for _ in classes]
    assert rows and len(rows) == d['dataset_counts']['test'], 'CSV ve test boyutu tutmuyor'
    hashes = set()
    accurate = 0
    for i, r in enumerate(rows):
        assert int(r['test_indeksi']) == i, f'Sıra tutarsız: {i}'
        assert r['gercek'] in classes and r['tahmin'] in classes
        actual = (r['gercek'] == r['tahmin'])
        assert r['dogru'] in ('True','False') and (r['dogru'] == 'True') == actual
        probs = [float(r['p_'+name]) for name in classes]
        assert all(0 <= p <= 1 for p in probs), f'Geçersiz olasılık: {i}'
        assert abs(sum(probs) - 1) < .001, f'Olasılıklar 1 toplamı vermiyor: {i}'
        assert abs(float(r['guven']) - probs[classes.index(r['tahmin'])]) < 1e-5
        assert probs.index(max(probs)) == classes.index(r['tahmin'])
        assert len(r['sha256']) == 64 and all(x in '0123456789abcdef' for x in r['sha256'])
        hashes.add(r['sha256'])
        cm[classes.index(r['gercek'])][classes.index(r['tahmin'])] += 1
        accurate += int(actual)
    assert cm == d['confusion_matrix'], 'Karmaşıklık matrisi CSV ile aynı değil'
    assert abs(accurate / len(rows) - float(d['metrics']['accuracy'])) < 1e-8
    recalls=[]; precisions=[];f1s=[]
    for j in range(len(classes)):
        tp = cm[j][j]
        support = sum(cm[j]); predicted = sum(cm[i][j] for i in range(len(classes)))
        rec = tp/support if support else 0
        prec = tp/predicted if predicted else 0
        f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
        precisions.append(prec); recalls.append(rec);f1s.append(f1)
    for name, values in [('precision_macro', precisions), ('recall_macro', recalls), ('f1_macro', f1s)]:
        assert abs(sum(values)/len(values) - d['metrics'][name]) < 1e-5, f'{name} tutarsız'
    by_index = {int(r['test_indeksi']):r for r in rows}
    for s in d['selected_samples']:
        r = by_index[int(s['test_indeksi'])]
        for field in ('sha256','dosya','gercek','tahmin'):
            assert str(s[field]) == r[field], f'Seçili örnek uyuşmazlığı: {field}'
        assert bool(s['dogru']) == (r['dogru'] == 'True')
        assert abs(float(s['guven'])-float(r['guven']))<1e-6
        p = Path(s['dosya'])
        if p.is_file():
            assert hashlib.sha256(p.read_bytes()).hexdigest() == s['sha256'], f'Seçilen resim farklı: {p}'
    right = sum(bool(s['dogru']) for s in d['selected_samples'])
    wrong = len(d['selected_samples']) - right
    assert right <= 3 and wrong <= 3
    for file in ('coffee_bean_cnn_model.keras','class_names.json','accuracy_loss_grafikleri.png',
                 'confusion_matrix.png','ornek_tahminler.png'):
        assert (out/file).is_file() and (out/file).stat().st_size > 0, f'Çıktı eksik: {file}'
    saved_classes=json.loads((out/'class_names.json').read_text(encoding='utf-8'))
    assert saved_classes == classes, 'Model sınıf sırası uyuşmuyor'
    print(f'✅ CSV+JSON+matris+F1+Softmax+SHA256 uyumlu. Test: {len(rows)}, doğru: {accurate}, yanlış: {len(rows)-accurate}, seçilen: {right} doğru + {wrong} yanlış.')
    return True


if __name__ == '__main__':
    check(sys.argv[1] if len(sys.argv)>1 else None)
