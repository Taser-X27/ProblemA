"""直接使用给定的40类固定权重融合ViT/CNN概率；不拟合或搜索系数。"""
import argparse
import csv
import json
import logging
from pathlib import Path

from training.common import ROOT, data_root, read, save, sha
import numpy as np

LOG = logging.getLogger('classwise_fusion')
SPLITS = ('validation', 'test')
MODELS = ('vit', 'cnn')
FORMULA = 's[k]=lambda[k]*p_vit[k]+(1-lambda[k])*p_cnn[k]; p=s/s.sum()'
WEIGHT_PATH = ROOT / 'configs/fusion.json'
WEIGHT_CONFIG = read(WEIGHT_PATH)
CLASSES = tuple(WEIGHT_CONFIG['classes'])
VIT_WEIGHTS = tuple(WEIGHT_CONFIG['vit_weights'])
CNN_WEIGHTS = tuple(WEIGHT_CONFIG['cnn_weights'])
if (len(CLASSES) != 40 or len(set(CLASSES)) != 40
        or np.asarray(VIT_WEIGHTS).shape != (40,) or np.asarray(CNN_WEIGHTS).shape != (40,)
        or not np.isfinite([VIT_WEIGHTS, CNN_WEIGHTS]).all()
        or (np.asarray([VIT_WEIGHTS, CNN_WEIGHTS]) < 0).any()
        or (np.asarray([VIT_WEIGHTS, CNN_WEIGHTS]) > 1).any()
        or not np.allclose(np.asarray(VIT_WEIGHTS) + CNN_WEIGHTS, 1, rtol=0, atol=1e-12)):
    raise ValueError('固定权重表必须包含40类，两路权重均在[0,1]内且每类之和为1')


def normalized_fusion(vit, cnn):
    """按固定类别顺序融合[N,40]概率；不使用样本标签，返回归一化概率。"""
    vit, cnn = np.asarray(vit, dtype=np.float64), np.asarray(cnn, dtype=np.float64)
    if vit.shape != cnn.shape or vit.ndim != 2 or vit.shape[1] != 40:
        raise ValueError('两模型概率必须具有相同的[N,40]形状')
    for probabilities in (vit, cnn):
        if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
                or not np.allclose(probabilities.sum(1), 1, rtol=0, atol=1e-6)):
            raise ValueError('每路输入必须为有限非负概率，且每行总和为1')
    weights = np.asarray(VIT_WEIGHTS)
    scores = weights * vit + (1 - weights) * cnn
    totals = scores.sum(1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError('融合结果包含零概率总和，无法归一化')
    return scores / totals


def metrics(labels, probabilities, classes):
    prediction = probabilities.argmax(1)
    confusion = np.zeros((40, 40), dtype=np.int64)
    np.add.at(confusion, (labels, prediction), 1)
    support, predicted, tp = confusion.sum(1), confusion.sum(0), confusion.diagonal()
    recall = np.divide(tp, support, out=np.zeros(40), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros(40), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(40), where=precision + recall > 0)
    return dict(samples=len(labels), correct=int(tp.sum()), accuracy=float(tp.sum() / len(labels)),
        macro_f1=float(f1.mean()), balanced_accuracy=float(recall.mean()),
        confusion_matrix_true_rows_predicted_columns=confusion.tolist(),
        per_class=[dict(class_id=k, name=name, support=int(support[k]), correct=int(tp[k]),
                        precision=float(precision[k]), recall=float(recall[k]), f1=float(f1[k]))
                   for k, name in enumerate(classes)])


def load_inputs(root, paths, splits=SPLITS):
    """核对manifest/provenance、类别顺序、checkpoint身份及每条预测，再按ID对齐。"""
    classes, rows, arrays, fingerprints, sources = None, [], {m: [] for m in MODELS}, {}, {}
    seen_ids, seen_clips, checkpoint_ids = set(), set(), {}
    subjects_by_split = {}
    for split in splits:
        manifest_path, provenance_path = [root / split / name for name in ('manifest.json', 'provenance.json')]
        manifest, provenance = read(manifest_path), read(provenance_path)
        fingerprints.update({str(p): sha(p) for p in (manifest_path, provenance_path)})
        if classes is None:
            classes = manifest['classes']
        if classes != manifest['classes'] or tuple(classes) != CLASSES:
            raise ValueError('数据集类别顺序必须与configs/fusion.json的40类权重表一致')
        expected = {r['file']: r for r in manifest['samples']}
        origin = {r['file']: r for r in provenance['samples']}
        if not expected or len(expected) != len(manifest['samples']) or set(expected) != set(origin) or len(origin) != len(provenance['samples']):
            raise ValueError(f'{split}: manifest/provenance为空、重复或样本不一致')
        clips = [r['source_clip'] for r in origin.values()]
        if seen_ids.intersection(expected) or seen_clips.intersection(clips) or len(set(clips)) != len(clips):
            raise ValueError('validation/test包含重复视频ID或source_clip')
        seen_ids.update(expected)
        seen_clips.update(clips)
        subjects_by_split[split] = sorted({r['subject'] for r in origin.values()})
        loaded = {}
        for model in MODELS:
            path = paths[f'{model}_{split}'].resolve()
            config_path = path.with_name('config.json')
            if not path.is_file() or not config_path.is_file():
                raise FileNotFoundError(f'缺少{path}或同目录config.json；请先用evaluate.py导出{model}/{split}概率')
            config = read(config_path)
            if config.get('model') != model or config.get('split') != split or config.get('classes') != classes:
                raise ValueError(f'{config_path}: 模型、split或类别顺序不匹配；请用新版evaluate.py重新导出')
            digest = sha(path)
            if config.get('prediction_sha256') != digest:
                raise ValueError(f'{path}: 预测文件哈希与config.json不一致')
            checkpoint = config.get('checkpoint_sha256')
            if not isinstance(checkpoint, str) or len(checkpoint) != 64:
                raise ValueError(f'{config_path}: 缺少checkpoint_sha256')
            identity = (checkpoint, config['epoch'])
            if model in checkpoint_ids and checkpoint_ids[model] != identity:
                raise ValueError(f'{model}: validation和test必须使用同一个checkpoint')
            checkpoint_ids[model] = identity
            indexed = {}
            for line in path.read_text(encoding='utf8').splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                sid = row['sample_id']
                if sid in indexed or sid not in expected:
                    raise ValueError(f'{path}: 重复或未知样本{sid}')
                label = expected[sid]['label']
                info = origin[sid]
                if not isinstance(label, int) or not 0 <= label < 40 or info['label'] != label:
                    raise ValueError(f'{split}/{sid}: 数据标签不合法或不一致')
                if (row['label'] != label or row['subject'] != info['subject'] or row['true_class'] != classes[label]
                        or row.get('split') != split or row.get('available_frames') != info['published_frames']
                        or row.get('checkpoint_epoch') != config['epoch']):
                    raise ValueError(f'{path}: {sid}的标签、被试、帧数或checkpoint信息不匹配')
                q = np.asarray(row['probabilities'], dtype=np.float64)
                if (q.shape != (40,) or not np.isfinite(q).all() or (q < 0).any()
                        or not np.isclose(q.sum(), 1, rtol=0, atol=1e-6)):
                    raise ValueError(f'{path}: {sid}必须有40维有效概率，总和为1')
                if row['prediction'] != int(q.argmax()) or row.get('answer') != classes[int(q.argmax())]:
                    raise ValueError(f'{path}: {sid}的预测类别与概率不一致')
                indexed[sid] = q
            if indexed.keys() != expected.keys():
                raise ValueError(f'{path}: 预测未覆盖整个{split}集')
            loaded[model] = indexed
            key = f'{model}_{split}'
            sources[key] = dict(predictions=str(path), **config)
            fingerprints.update({str(path): digest, str(config_path): sha(config_path)})
        for sid in sorted(expected):
            info = origin[sid]
            rows.append(dict(sample_id=sid, split=split, label=expected[sid]['label'], subject=info['subject'],
                source_clip=info['source_clip'], available_frames=info['published_frames']))
            for model in MODELS:
                arrays[model].append(loaded[model][sid])
    return classes, rows, {m: np.asarray(q) for m, q in arrays.items()}, fingerprints, sources, subjects_by_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--split', choices=['combined', 'validation', 'eval', 'test'], default='combined',
                        help='选择融合数据集；默认combined合并validation和test；eval为validation别名')
    for model in MODELS:
        for split in SPLITS:
            flags = [f'--{model}-{split}']
            if split == 'validation':
                flags.append(f'--{model}-eval')
            parser.add_argument(*flags, dest=f'{model}_{split}', type=Path,
                default=ROOT / f'results/eval_{model}_{split}/predictions.jsonl')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--check', action='store_true', help='只核对输入与固定权重，不写结果')
    args = parser.parse_args()
    if args.split == 'eval':
        args.split = 'validation'
    splits = SPLITS if args.split == 'combined' else (args.split,)
    suffix = 'validation_test' if args.split == 'combined' else args.split
    out = (args.output_dir or ROOT / f'results/fusion_fixed_{suffix}').resolve()
    if not out.is_relative_to(ROOT / 'results') or out == ROOT / 'results':
        parser.error('请使用本工程results下的独立输出目录')
    paths = {f'{m}_{s}': getattr(args, f'{m}_{s}') for m in MODELS for s in splits}
    # 确认本进程中加载的权重表与磁盘一致。
    weight_digest = sha(WEIGHT_PATH)
    if read(WEIGHT_PATH) != WEIGHT_CONFIG:
        raise RuntimeError('固定权重表在加载后发生变化')
    classes, rows, q, protected, sources, subjects = load_inputs(data_root(args.data_root), paths, splits)
    protected[str(Path(__file__).resolve())] = sha(Path(__file__))
    protected[str(WEIGHT_PATH)] = weight_digest
    groups = {s: sum(r['split'] == s for r in rows) for s in splits}
    fused = normalized_fusion(q['vit'], q['cnn'])
    if args.check:
        print(json.dumps(dict(passed=True, samples=len(rows), groups=groups, classes=len(classes),
                              weights_source=str(WEIGHT_PATH), fitting_started=False)))
        return
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('已有结果受保护；请指定新的--output-dir')
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', force=True,
        handlers=[logging.StreamHandler(), logging.FileHandler(out / 'fusion.log', encoding='utf8')])
    LOG.info('Applying fixed 40-class weights from %s: %s', WEIGHT_PATH, groups)
    # 标签只用于计算指标，不用于融合或选择权重。
    labels = np.asarray([r['label'] for r in rows], dtype=np.int64)
    methods = dict(vit=q['vit'], cnn=q['cnn'], classwise_fixed=fused)
    partitions = dict(combined=np.ones(len(rows), dtype=bool),
        **{s: np.asarray([r['split'] == s for r in rows]) for s in splits})
    scores = {part: {name: metrics(labels[mask], probabilities[mask], classes) for name, probabilities in methods.items()}
              for part, mask in partitions.items()}
    per_subject = {s: metrics(labels[mask], fused[mask], classes)
        for s in sorted({r['subject'] for r in rows})
        for mask in [np.asarray([r['subject'] == s for r in rows])]}
    selection = dict(formula=FORMULA, lambda_by_class=list(VIT_WEIGHTS), cnn_weights=list(CNN_WEIGHTS),
        classes=classes, weights_source=str(WEIGHT_PATH), weight_metadata=WEIGHT_CONFIG,
        fitting_performed=False, independent_evaluation=False,
        note='直接使用用户给定图表的两位小数权重；原权重在validation+test上拟合，这两个集合的成绩不属于独立测试成绩。',
        samples=len(rows), groups=groups, subjects=subjects, source_predictions=sources, input_sha256=protected)
    save(out / 'selection.json', selection)
    save(out / 'metrics.json', scores)
    save(out / 'per_subject.json', per_subject)
    predictions = []
    for row, probabilities in zip(rows, fused):
        pred = int(probabilities.argmax())
        predictions.append(dict(row, true_class=classes[row['label']], prediction=pred, answer=classes[pred],
            correct=pred == row['label'], confidence=float(probabilities[pred]), probabilities=probabilities.tolist()))
    for part, mask in partitions.items():
        target = out / f'predictions_{part}.jsonl'
        selected = [r for r, keep in zip(predictions, mask) if keep]
        target.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected), encoding='utf8')
        restored = [json.loads(line) for line in target.read_text(encoding='utf8').splitlines()]
        restored_q = np.asarray([r['probabilities'] for r in restored])
        if (not np.allclose(restored_q, fused[mask], rtol=0, atol=1e-15)
                or not np.allclose(restored_q.sum(1), 1)
                or int((restored_q.argmax(1) == labels[mask]).sum()) != scores[part]['classwise_fixed']['correct']):
            raise RuntimeError('保存的融合概率复核失败')
    with (out / 'lambda_by_class.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['class_id', 'class_name', 'vit_lambda', 'cnn_weight'])
        for k in range(40):
            writer.writerow([k, classes[k], VIT_WEIGHTS[k], CNN_WEIGHTS[k]])
    lines = ['# ViT + CNN：40类固定权重概率融合', '',
        f'样本：{len(rows)}；各集合：{groups}。本次未拟合或搜索任何系数。',
        '权重来自用户给定图表，直接使用图中两位小数。原权重在validation+test上拟合，这两个集合不属于独立测试集。', '',
        f'公式：`{FORMULA}`。λ是ViT权重，1−λ是CNN权重。',
        '40个系数用于所有视频的40个输出列；不根据视频真实标签选择系数。', '',
        '|方法|' + '|'.join(partitions) + '|', '|---|' + '---:|' * len(partitions)]
    for method in methods:
        cells = [scores[p][method] for p in partitions]
        lines.append('|' + method + '|' + '|'.join(f'{m["correct"]}/{m["samples"]} ({m["accuracy"]:.2%})' for m in cells) + '|')
    lines += ['', '40类固定系数见lambda_by_class.csv及selection.json；逐视频融合概率见predictions_*.jsonl。',
              'selection.json记录类别顺序、权重来源、两模型checkpoint哈希及全部输入哈希。']
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    for path, digest in protected.items():
        if sha(path) != digest:
            raise RuntimeError(f'融合期间输入发生变化：{path}')
    save(out / 'verification.json', dict(passed=True, samples=len(rows), groups=groups,
        ids_labels_subjects_frames_classes_and_checkpoint_identity_checked=True,
        original_inputs_unchanged=True, written_predictions_rescored=True,
        fitting_performed=False, independent_evaluation=False))
    print(json.dumps(dict(output=str(out), samples=len(rows), fitting_performed=False,
        results={p: {name: {k: m[k] for k in ('samples', 'correct', 'accuracy')} for name, m in group.items()}
                 for p, group in scores.items()})))


if __name__ == '__main__':
    main()
