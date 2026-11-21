# 由prepare_cnn.py启动：新数据索引、新YOLO26n缓存；裁剪与解码规则沿用原实现。
"""为train/validation或独立的test构建Depth_Color RGB+IR人物/手部缓存。"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gc
import logging
from pathlib import Path
import time

from .common import CACHE, PERSON, POSE, data_root, read, save, sha, inventory, verify_upstream, check_setup
from .cnn_cache import check_identity, records_digest, save_identity
import numpy as np
from PIL import Image
import torch
from sota_data import files, window
from sota_joint import pose_probes, hand_crop, POSE as POSE_CONFIG, CROP
from .cnn_input import decode_depth_ir, CHANNEL_ORDER

LOG=logging.getLogger('prepare')
MODALITY='depth_ir'  # 固定Depth_Color RGB＋IR，人物与手部均使用双模态。


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--split',choices=['trainval','test'],default='trainval',
                        help='trainval写cache/cnn；test单独写cache/cnn_test')
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if args.check:
        import json
        result=check_setup('cnn',args.data_root)
        if args.split=='test':result['test']=len(inventory(args.data_root,('test',))[0]['test'])
        print(json.dumps(result));return
    cache=CACHE if args.split=='trainval' else CACHE.with_name(CACHE.name+'_test')
    cache.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[
        logging.StreamHandler(),logging.FileHandler(cache/'prepare.log',encoding='utf8')])
    root=data_root(args.data_root)
    split,classes=inventory(root,('train','validation') if args.split=='trainval' else ('test',))
    identity=dict(records_sha256=records_digest(split),
        person_sha256=sha(PERSON),pose_sha256=sha(POSE),frames=32,size=128,
        person_detector='YOLO26n (project configuration)',wrist_detector='YOLO26n-pose (requested replacement)',
        person_rule='8 probes, conf0.25, normalized union, margin1.4, min_side0.35*width, clipped',
        pose=dict(POSE_CONFIG),hand_crop=dict(CROP),
        split_samples={name:len(rows) for name,rows in split.items()},modalities=['Depth_Color','IR'],source_manifest=verify_upstream())
    if MODALITY=='depth_ir':
        identity.update(input_mode='depth_ir',modalities=['Depth_Color','IR'],channel_order=CHANNEL_ORDER,
            pairing='identical timestamp and frame number; identical pixel crop',
            depth_representation='published Depth_Color RGB, no scalar conversion',
            normalization='original Kinetics RGB mean/std; IR uses channel means')
    complete_exists=(cache/'complete.json').exists()
    rewrites={}
    if complete_exists or (cache/'identity.json').exists():
        rewrites=check_identity(cache,identity,split,root,complete_exists)
    if complete_exists:
        complete=read(cache/'complete.json')
        expected_arrays={f'{name}_{kind}.npy' for name in split for kind in ('body','hands')}
        hashes=complete['cache_sha256']
        if not complete['passed'] or set(hashes)!=expected_arrays:
            raise ValueError(f'{cache}：缓存完成标记或数组列表不完整')
        for name,digest in hashes.items():
            if sha(cache/name)!=digest:
                raise ValueError(f'{cache/name}：缓存数组校验失败')
        save_identity(cache,identity,split,rewrites)
        if rewrites:
            LOG.info('Migrated existing cache records to dataset-relative paths')
        LOG.info('Verified existing complete cache')
        return
    save_identity(cache,identity,split,rewrites)
    windows_path=cache/'windows.json'
    windows=read(windows_path) if windows_path.exists() else {}
    records=[r for rows in split.values() for r in rows]
    todo=[r for r in records if r['sample_id'] not in windows]
    if todo:
        from ultralytics import YOLO
        torch.set_num_threads(2)
        person=YOLO(str(PERSON))
        pose=YOLO(str(POSE))
        runtime=dict(person_parameters=sum(p.numel() for p in person.model.parameters()),
                     pose_parameters=sum(p.numel() for p in pose.model.parameters()),
                     pose_task=pose.task,person_task=person.task)
        assert pose.task=='pose' and person.task=='detect'
        save(cache/'detector_parameters.json',runtime)
        start=time.perf_counter()
        for i,r in enumerate(todo):
            ir_folder=root/r['folders']['IR']
            paths=files(ir_folder)
            indices=sorted(set(np.linspace(0,len(paths)-1,8).round().astype(int)))
            images=[]
            for j in indices:
                with Image.open(paths[j]) as im:
                    images.append(np.asarray(im.convert('RGB')))
            results=person.predict(images,classes=[0],conf=.25,imgsz=640,device=0,verbose=False,quantize=32)
            boxes=[]
            for res in results:
                if len(res.boxes):
                    h,w=res.orig_shape
                    boxes.append((res.boxes.xyxy[res.boxes.conf.argmax()].cpu().numpy()/[w,h,w,h]).tolist())
            body=window(boxes,'union')
            kp,size=pose_probes(pose,ir_folder,body)
            hands,status=hand_crop(kp,body,size)
            windows[r['sample_id']]=dict(body=body,hands=hands,status=status,shape_wh=list(size),
                person_boxes=boxes,confident_wrist_observations=int((np.nan_to_num(kp[:,[9,10],2])>=.5).sum()))
            if (i+1)%20==0 or i+1==len(todo):
                save(windows_path,windows)
                elapsed=time.perf_counter()-start
                LOG.info('windows %d/%d, %.2f clips/s, ETA %.1fmin; hand routes=%s',len(windows),len(records),
                    (i+1)/elapsed,(len(todo)-i-1)*elapsed/(i+1)/60,dict(Counter(x['status'] for x in windows.values())))
        del person,pose
        gc.collect();torch.cuda.empty_cache()
    for name,records in split.items():
        save(cache/f'{name}_records.json',records)
        state_path=cache/f'{name}_decode_progress.json'
        state=read(state_path) if state_path.exists() else dict(decoded=0,complete=False)
        if state['complete']:
            continue
        partial={kind:cache/f'{name}_{kind}.partial.npy' for kind in ['body','hands']}
        # Recover a crash between the two final renames without recreating a completed array.
        if int(state['decoded'])==len(records):
            for kind,path in partial.items():
                final=cache/f'{name}_{kind}.npy'
                assert path.exists() or final.exists(),f'Missing decoded cache: {kind}'
                if path.exists():path.replace(final)
            save(state_path,dict(decoded=len(records),complete=True))
            continue
        arrays={}
        for kind,path in partial.items():
            arrays[kind]=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',
                        dtype=np.uint8,shape=(len(records),32,4 if MODALITY=='depth_ir' else 1,128,128))
        start=int(state['decoded'])
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for begin in range(start,len(records),32):
                end=min(begin+32,len(records))
                jobs=records[begin:end]
                decoded=pool.map(lambda r:decode_depth_ir(r,windows[r['sample_id']],root),jobs)
                for index,(body,hands) in enumerate(decoded,start=begin):
                    arrays['body'][index]=body;arrays['hands'][index]=hands
                for array in arrays.values():
                    array.flush()
                save(state_path,dict(decoded=end,complete=False))
                if begin%128==0 or end==len(records):
                    LOG.info('%s cache %d/%d',name,end,len(records))
        for mapped in arrays.values():
            mapped.flush();mapped._mmap.close()
        del mapped,arrays
        gc.collect()
        for kind,path in partial.items():
            path.replace(cache/f'{name}_{kind}.npy')
        save(state_path,dict(decoded=len(records),complete=True))
    hashes={f'{s}_{kind}.npy':sha(cache/f'{s}_{kind}.npy') for s in split for kind in ['body','hands']}
    summary={s:dict(Counter(windows[r['sample_id']]['status'] for r in records)) for s,records in split.items()}
    save(cache/'classes.json',classes)
    save(cache/'complete.json',dict(passed=True,cache_sha256=hashes,hand_routes=summary,
         samples=sum(len(v) for v in split.values()),IR_only=MODALITY=='ir',input_mode=MODALITY,original_caches_untouched=True))
    LOG.info('CACHE COMPLETE: %s',summary)


if __name__=='__main__':
    main()
