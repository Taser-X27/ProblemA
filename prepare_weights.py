"""下载公开预训练权重，提取 ViT-H 骨干，并校验文件的 SHA-256。"""
import argparse
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

from training.common import ASSETS, ROOT, sha


def verify(path, expected):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f'{path.name} SHA-256 不匹配\n预期：{expected}\n实际：{actual}')


def download(url, path):
    request = Request(url, headers={'User-Agent': 'ProblemA-weight-setup', 'Accept-Encoding': 'identity'})
    with urlopen(request, timeout=90) as response, path.open('wb') as stream:
        shutil.copyfileobj(response, stream, length=1024 * 1024)


def extract_teacher(source, output, original_sha256):
    import torch

    checkpoint = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
    backbone = {name.removeprefix('backbone.'): value
                for name, value in checkpoint['state_dict'].items()
                if name.startswith(('backbone.patch_embed.', 'backbone.blocks.', 'backbone.fc_norm.'))}
    if len(backbone) != 420:
        raise ValueError('官方 ViT-H checkpoint 的骨干张量数量不匹配')
    torch.save(dict(model=backbone, source_sha256=original_sha256), output)


def prepare_asset(name, asset, root, teacher_source=None, check=False):
    output = root / asset['file']
    if output.exists() or check:
        verify(output, asset['sha256'])
        print(f'已校验 {output.name}', flush=True)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.prepare-', dir=output.parent) as folder:
        # torch.save 把文件名写入归档，临时文件使用与目标相同的文件名。
        candidate = Path(folder) / output.name
        if name == 'teacher':
            source = teacher_source or (Path(folder) / 'pytorch_model.bin')
            if teacher_source is None:
                print('正在下载 ViT-H 官方 checkpoint（约 5.05 GB）…', flush=True)
                download(asset['source'], source)
            verify(source, asset['original_sha256'])
            print('正在提取 ViT-H 骨干…', flush=True)
            extract_teacher(source, candidate, asset['original_sha256'])
        else:
            print(f'正在下载 {output.name}…', flush=True)
            download(asset['download_url'] if 'download_url' in asset else asset['source'], candidate)
        verify(candidate, asset['sha256'])
        candidate.replace(output)
    print(f'已校验并保存 {output.name}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='仅校验已准备好的权重')
    parser.add_argument('--teacher-source', type=Path, help='使用已下载的官方 ViT-H pytorch_model.bin')
    args = parser.parse_args()
    try:
        for name, asset in ASSETS.items():
            prepare_asset(name, asset, ROOT, args.teacher_source, args.check)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f'准备权重失败：{error}\n')


if __name__ == '__main__':
    main()
