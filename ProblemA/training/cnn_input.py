"""CNN双模态输入：Depth_Color RGB在前，IR灰度在后；训练/推理共用解码。"""
from pathlib import Path

import numpy as np
from PIL import Image
from .common import resolve_folders

CHANNEL_ORDER = ['Depth_R', 'Depth_G', 'Depth_B', 'IR']


def paired_paths(folders):
    """核对传感器文件名中的时间戳及帧号，避免仅按数量配对。"""
    paths = {m: sorted(Path(folders[m]).glob('*.png')) for m in ['Depth_Color', 'IR']}
    if not paths['IR'] or len(paths['IR']) != len(paths['Depth_Color']):
        raise ValueError(f'IR/Depth帧数不匹配或为空：{folders}')
    def key(path, modality):
        prefix, suffix = ('Depth_', '_Color.png') if modality == 'Depth_Color' else ('IR_', '.png')
        name = path.name
        if not (name.startswith(prefix) and name.endswith(suffix)):
            raise ValueError(f'无法识别帧文件名：{path}')
        return name[len(prefix):-len(suffix)]
    if [key(p, 'Depth_Color') for p in paths['Depth_Color']] != [key(p, 'IR') for p in paths['IR']]:
        raise ValueError(f'IR/Depth时间戳或帧号不一致：{folders}')
    return paths


def decode_depth_ir(record, windows, root):
    """返回人物/手部两份uint8 [32,4,128,128]；不反解或灰度化Depth_Color。"""
    paths = paired_paths(resolve_folders(root, record['folders']))
    if 'frames' in record and len(paths['IR']) != record['frames']:
        raise ValueError('帧数与样本索引不一致')
    picks = np.linspace(0, len(paths['IR']) - 1, 32).round().astype(int)
    decoded = {}
    for j in np.unique(picks):
        with Image.open(paths['Depth_Color'][j]) as image:
            depth = image.convert('RGB')
        with Image.open(paths['IR'][j]) as image:
            ir = image.convert('L')
        if depth.size != ir.size:
            raise ValueError(f"两模态图像尺寸不一致：{record.get('sample_id')}")
        decoded[int(j)] = depth, ir
    result = []
    for kind in ['body', 'hands']:
        frames = []
        for j in picks:
            depth, ir = decoded[int(j)]
            crop = windows[kind]
            if crop is not None:
                w, h = ir.size
                box = tuple(round(v * s) for v, s in zip(crop, [w, h, w, h]))
                depth, ir = depth.crop(box), ir.crop(box)
            depth = np.asarray(depth.resize((128, 128), Image.Resampling.BILINEAR))
            ir = np.asarray(ir.resize((128, 128), Image.Resampling.BILINEAR))
            frames.append(np.concatenate([depth, ir[..., None]], axis=2).transpose(2, 0, 1))
        result.append(np.stack(frames))
    return result
