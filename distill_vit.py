"""先冻结ViT-H，对ViT-S＋Linear进行逐token蒸馏；不训练分类头。"""
from training.distill_engine import main

if __name__=='__main__':
    __import__('multiprocessing').freeze_support()
    main()
