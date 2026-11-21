"""先运行distill_vit.py；再加载本次最佳S，接随机MLP，全参数分类训练。"""
from training.vit_engine import main

if __name__=='__main__':
    __import__('multiprocessing').freeze_support()
    main()
