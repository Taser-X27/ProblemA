"""校验 CNN 缓存索引，并兼容旧版 Windows/Linux 绝对路径。"""
import hashlib
import json
from pathlib import PurePosixPath, PureWindowsPath

from .common import read, resolve_folders, sample_folders, save


def records_digest(records):
    return hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()


def portable_records(records):
    """只转换旧绝对路径，检查路径确实指向该条记录的集合、样本和模态。"""
    result = []
    for row in records:
        folders = row['folders']
        expected = sample_folders(row['split'],row['sample_id'])
        if folders.keys() != expected.keys():
            raise ValueError(f"缓存模态不一致：{row['sample_id']}")
        for modality, relative in expected.items():
            path = folders[modality]
            if path == relative:
                continue
            windows_path = PureWindowsPath(path)
            old = windows_path if windows_path.is_absolute() else PurePosixPath(path)
            suffix = PurePosixPath(relative).parts
            if not old.is_absolute() or old.parts[-len(suffix):] != suffix:
                raise ValueError(f'缓存样本路径不一致：{path}')
        result.append(dict(row,folders=expected))
    return result


def verify_records(path, expected):
    stored = read(path)
    if portable_records(stored) != expected:
        raise ValueError(f'{path}：缓存样本、顺序或帧数与当前数据不一致')
    return stored


def check_identity(cache, identity, indexed, root, complete):
    """校验旧身份与各索引，返回需要重新保存的记录。"""
    previous = read(cache/'identity.json')
    old_digest = previous.pop('records_sha256')
    if previous != {k:v for k,v in identity.items() if k != 'records_sha256'}:
        raise ValueError('缓存检测器或预处理配置与当前项目不一致')
    stored = {}
    for split, rows in indexed.items():
        path = cache/f'{split}_records.json'
        if complete or path.exists():
            stored[split] = verify_records(path,rows)
    if old_digest != identity['records_sha256']:
        if stored.keys() == indexed.keys() and records_digest(stored) == old_digest:
            return stored
        if not complete:
            # 旧版可能在写出任何索引前中断。同一路径下可由当前索引重建旧哈希。
            legacy = {split: [dict(row,folders={m:str(p) for m,p in
                       resolve_folders(root,row['folders']).items()}) for row in rows]
                      for split,rows in indexed.items()}
            if records_digest(legacy) == old_digest:
                return stored
        raise ValueError('缓存索引哈希不匹配。未写完索引的旧缓存须先在原数据路径下恢复。')
    return {split:rows for split,rows in stored.items() if rows != indexed[split]}


def save_identity(cache, identity, indexed, rewrites):
    # 先提交新身份。若进程在更新记录间中断，新身份也能核验并接受其余旧记录。
    path = cache/'identity.json'
    if not path.exists() or read(path) != identity:
        save(path,identity)
    for split in rewrites:
        save(cache/f'{split}_records.json',indexed[split])
