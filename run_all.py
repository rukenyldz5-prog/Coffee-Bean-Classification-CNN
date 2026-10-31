"""Tek komutta deneyin yeniden çalıştırılması ve raporun sonuçlarla güncellenmesi."""
from pathlib import Path
import subprocess
import sys
import zipfile
import json

ROOT=Path(__file__).resolve().parent

def run(args):
    print('\nÇalıştırılıyor:', ' '.join(map(str,args)),flush=True)
    subprocess.run(args,cwd=ROOT,check=True)

def main():
    run([sys.executable,str(ROOT/'train.py')])
    run([sys.executable,str(ROOT/'deneyi_dogrula.py')])
    run([sys.executable,str(ROOT/'raporu_tamamla.py')])
    output=ROOT/'artifacts'
    required=[
        'LBLM368_Ruken_YILDIZ_GERCEK_SONUCLAR_RAPORU.docx',
        'accuracy_loss_grafikleri.png','confusion_matrix.png','ornek_tahminler.png',
        'sonuclar.json','tekil_test_tahminleri.csv','coffee_bean_cnn_model.keras','class_names.json'
    ]
    for r in required:
        if not (output/r).is_file():
            raise FileNotFoundError(f'Gerçek deney çıktı dosyası eksik: {r}')
    with (output/'sonuclar.json').open(encoding='utf-8') as f:d=json.load(f)
    test_count=sum(sum(row) for row in d['confusion_matrix'])
    assert test_count==d['dataset_counts']['test']
    doc=output/'LBLM368_Ruken_YILDIZ_GERCEK_SONUCLAR_RAPORU.docx'
    pdf=doc.with_suffix('.pdf')
    final_zip=ROOT/'LBLM368_Ruken_YILDIZ_GERCEK_DENEY_TESLIM_PAKETI.zip'
    with zipfile.ZipFile(final_zip,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=7) as z:
        for name in ('train.py','predict.py','requirements.txt','README.md','raporu_tamamla.py','run_all.py','deneyi_dogrula.py','github_guncelle.py'):
            z.write(ROOT/name,arcname=name)
        for artifact in output.iterdir():
            if artifact.is_file(): z.write(artifact,arcname='artifacts/'+artifact.name)
    print('\n✅ Gerçek test doğruluğu:',d['metrics']['accuracy'])
    print('✅ Gerçek tahminler CSV:',output/'tekil_test_tahminleri.csv')
    print('✅ Word raporu:',doc)
    print('✅ PDF raporu:' if pdf.exists() else 'ℹ PDF üretilemedi (LibreOffice gerekli):',pdf)
    print('✅ ZIP teslim paketi:',final_zip)

if __name__=='__main__':main()
