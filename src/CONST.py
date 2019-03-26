import os
import urllib.request
import zipfile
import shutil

INDIR = '../data/input'
INTRNDIR = '../data/input/train'
INTSTDIR = '../data/input/test'
OUTDIR = '../data/output'
IMPDIR = os.path.join(OUTDIR, 'imp')
PIPE000 = os.path.join(OUTDIR, '_000')
PIPE001 = os.path.join(OUTDIR, '_001_{}')
PIPE002 = os.path.join(OUTDIR, '_002')
PIPE003 = os.path.join(OUTDIR, '_003')
EX_COLS = ['Engine', 'RUL', 'EncodedEngine']

for _dir in [INDIR, OUTDIR, PIPE000, PIPE002, PIPE003]:
    if not os.path.exists(_dir):
        os.makedirs(_dir)

# For training data
if not os.path.exists(INTRNDIR):
    os.makedirs(INTRNDIR)
    url_train = "https://industrial-big-data.io/wp-content/themes/fcvanilla/DLdate/Train%20Files.zip"
    urllib.request.urlretrieve(url_train, os.path.join(INDIR, 'TrainFiles.zip'))
    with zipfile.ZipFile(os.path.join(INDIR, 'TrainFiles.zip')) as existing_zip:
        existing_zip.extractall(INTRNDIR)
    trn_files = os.listdir(os.path.join(INTRNDIR, 'Train Files'))
    for f in trn_files:
        shutil.move(os.path.join(INTRNDIR, 'Train Files', f), os.path.join(INTRNDIR, f))
    paths = os.listdir(INTRNDIR)
    for path in paths:
        if os.path.isdir(os.path.join(INTRNDIR, path)):
            shutil.rmtree(os.path.join(INTRNDIR, path))

# For test data
if not os.path.exists(INTSTDIR):
    os.makedirs(INTSTDIR)
    url_test = "https://industrial-big-data.io/wp-content/themes/fcvanilla/DLdate/Test%20Files.zip"
    urllib.request.urlretrieve(url_test, os.path.join(INDIR, 'TestFiles.zip'))
    with zipfile.ZipFile(os.path.join(INDIR, 'TestFiles.zip')) as existing_zip:
        existing_zip.extractall(INTSTDIR)
    trn_files = os.listdir(os.path.join(INTSTDIR, 'Test Files'))
    for f in trn_files:
        shutil.move(os.path.join(INTSTDIR, 'Test Files', f), os.path.join(INTSTDIR, f))
    paths = os.listdir(INTSTDIR)
    for path in paths:
        if os.path.isdir(os.path.join(INTSTDIR, path)):
            shutil.rmtree(os.path.join(INTSTDIR, path))

# For submit samples
if not os.path.exists(os.path.join(INDIR, 'submit_sample.csv')):
    ss_url = "https://industrial-big-data.io/wp-content/themes/fcvanilla/DLdate/csv.php"
    proxy = urllib.request.ProxyHandler({'https': 'http://autoproxy.kmt.kmtg.net:8080'})
    opener = urllib.request.build_opener(proxy)
    urllib.request.install_opener(opener)
    url_submit_sample = "https://industrial-big-data.io/wp-content/themes/fcvanilla/DLdate/Test%20Files.zip"
    urllib.request.urlretrieve(url_submit_sample, os.path.join(INDIR, 'submit_sample.csv'))