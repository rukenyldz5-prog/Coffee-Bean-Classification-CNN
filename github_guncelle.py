"""SADECE doğrulanmış gerçek deney sonrasında GitHub deposuna sonuçları yükler.
Erişim anahtarı asla bu dosyaya veya not defterine yazılmaz.
GITHUB_TOKEN, GitHub fine-grained token, Contents read/write gerekir.
"""
import base64
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parent
REPO='rukenyldz5-prog/Coffee-Bean-Classification-CNN'
BRANCH='main'


def api(method, rel, token, payload=None):
    url='https://api.github.com/repos/'+REPO+'/'+rel
    data=json.dumps(payload).encode('utf-8') if payload is not None else None
    req=Request(url,data=data,method=method,headers={
        'Authorization':'Bearer '+token,
        'Accept':'application/vnd.github+json',
        'X-GitHub-Api-Version':'2022-11-28',
        'User-Agent':'Ruken-Coffee-Research',
        **({'Content-Type':'application/json'} if data else {})})
    try:
        with urlopen(req,timeout=45) as res:return json.load(res)
    except HTTPError as e:
        body=e.read(2000).decode('utf-8','replace')
        raise RuntimeError(f'GitHub HTTP {e.code}: {body}') from None


def upload(path_in_repo, localfile, token):
    if not localfile.is_file():raise FileNotFoundError(localfile)
    url='contents/'+quote(path_in_repo,safe='/')
    try:
        sha=api('GET',url+'?ref='+BRANCH,token).get('sha')
    except RuntimeError as ex:
        if 'GitHub HTTP 404:' not in str(ex):raise
        sha=None
    data={'message':'LBLM368: doğrulanan deney ve güncel kaynak '+path_in_repo,
          'content':base64.b64encode(localfile.read_bytes()).decode('ascii'),
          'branch':BRANCH}
    if sha:data['sha']=sha
    api('PUT',url,token,data)
    print('✅ GitHub:',path_in_repo)


def main():
    token=os.environ.get('GITHUB_TOKEN','').strip()
    if not token:raise RuntimeError('GITHUB_TOKEN bulunamadı; GitHub hesabına yazma işlemi yapılmadı.')
    # Model yeniden eğitilmeden yanlışlıkla eski grafikleri yüklemeyelim.
    from deneyi_dogrula import check
    check()
    out=ROOT/'artifacts'
    paths={
      name:ROOT/name for name in ('train.py','predict.py','run_all.py','raporu_tamamla.py',
              'deneyi_dogrula.py','github_guncelle.py','README.md','requirements.txt',
              '.gitignore','rapor_sablonu.docx')}
    for name in ('accuracy_loss_grafikleri.png','confusion_matrix.png','ornek_tahminler.png'):
        paths[name]=out/name
    for name in ('tekil_test_tahminleri.csv','sonuclar.json','class_names.json',
                 'LBLM368_Ruken_YILDIZ_GERCEK_SONUCLAR_RAPORU.docx',
                 'LBLM368_Ruken_YILDIZ_GERCEK_SONUCLAR_RAPORU.pdf'):
        p=out/name
        if p.exists():paths['artifacts/'+name]=p
    if (ROOT/'LBLM368_CNN_SON_OTOMATIK_COLAB.ipynb').exists():
        paths['LBLM368_CNN_SON_OTOMATIK_COLAB.ipynb']=ROOT/'LBLM368_CNN_SON_OTOMATIK_COLAB.ipynb'
    if any(not p.exists() for p in paths.values()):
        raise FileNotFoundError('GitHub dosyalarının bazıları eksik; kaynakları ve deney sonuçlarını kontrol et.')
    # Önce erişim doğrulaması: hak olmayan hesapla işlem yapma.
    rep=api('GET','',token)
    if rep.get('full_name','').lower()!=REPO.lower():
        raise RuntimeError('Repo adı uyuşmuyor; yanlış depoya yükleme iptal edildi')
    print('Gerçek deney doğrulandı. GitHub deposuna yeni kaynak dosyaları ve raporlar yükleniyor...')
    for name,path in paths.items():upload(name,path,token)
    print('✅ Repo güncellendi. Model ağırlığı GitHub’a gönderilmedi; Colab ZIP paketinde bulunur.')

if __name__=='__main__':main()
