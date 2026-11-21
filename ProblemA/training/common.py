"""项目内路径、配置和数据索引；不引用工作区其他实验目录。"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys

ROOT=Path(__file__).resolve().parents[1]
VENDOR=ROOT/'vendor'
if sys.platform == 'win32' and sys.version_info[:2] == (3,12) and VENDOR.is_dir():
    sys.path.insert(0,str(VENDOR))
sys.path.insert(0,str(ROOT/'upstream/cnn'))
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'cache/yolo_settings'))
os.environ.setdefault('YOLO_AUTOINSTALL','false')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'cache/matplotlib'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    temp.replace(path)


ASSETS=read(ROOT/'configs/assets.json')
SETTINGS=read(ROOT/'configs/training.json')
PRETRAINED=ROOT/ASSETS['cnn']['file']
PRETRAINED_SHA=ASSETS['cnn']['sha256']
PERSON=ROOT/ASSETS['person']['file']
POSE=ROOT/ASSETS['pose']['file']
CACHE=ROOT/'cache/cnn'
RECIPE=SETTINGS['cnn']
CNN_INPUT_MODE='ir'
CNN_INPUT_CHANNELS=['IR']*4
MODALITIES=('Depth_Color','IR')


def data_root(override=None):
    value=Path(override) if override is not None else Path(read(ROOT/'configs/dataset.json')['data_root'])
    return value.resolve() if value.is_absolute() else (ROOT/value).resolve()


def verify_upstream():
    manifest=read(ROOT/'upstream/sources.json')
    for name,record in manifest.items():
        assert sha(ROOT/'upstream'/name)==record['sha256'],name
    return manifest


def sample_folders(split, sample_id):
    return {m: PurePosixPath(split, sample_id, m).as_posix() for m in MODALITIES}


def resolve_folders(root, folders):
    return {m: root / path for m,path in folders.items()}


def inventory(override=None,splits=('train','validation')):
    from .cnn_input import paired_paths
    root=data_root(override);classes=None;result={}
    for split in splits:
        manifest=read(root/split/'manifest.json')
        if classes is None:classes=manifest['classes']
        assert classes==manifest['classes'] and len(classes)==40
        provenance={r['file']:r for r in read(root/split/'provenance.json')['samples']}
        records=[]
        for row in manifest['samples']:
            p=provenance[row['file']]
            folders=sample_folders(split,row['file'])
            count=len(paired_paths(resolve_folders(root,folders))['IR'])
            assert count==p['published_frames']
            records.append(dict(sample_id=row['file'],split=split,label=row['label'],subject=p['subject'],
                source_clip=p['source_clip'],frames=count,folders=folders))
        result[split]=records
    if 'train' in result and 'validation' in result:
        assert not {r['subject'] for r in result['train']}&{r['subject'] for r in result['validation']}
        assert not {r['sample_id'] for r in result['train']}&{r['sample_id'] for r in result['validation']}
    return result,classes


def check_setup(model='all',override=None):
    split,classes=inventory(override)
    keys=['vit'] if model=='vit' else ['vit','teacher'] if model=='distill' else ['cnn','person','pose'] if model=='cnn' else list(ASSETS)
    for key in keys:
        info=ASSETS[key]
        assert sha(ROOT/info['file'])==info['sha256'],key
    verify_upstream()
    return dict(passed=True,model=model,data_root=str(data_root(override)),classes=len(classes),
        train=len(split['train']),validation=len(split['validation']),
        cnn_channels=CNN_INPUT_CHANNELS,training_started=False,optimizer_steps=0)
