"""从公开预训练初始化四通道IR CNN；--check只检查，不训练。"""
from training.cnn_engine import main

if __name__=='__main__':
    __import__('multiprocessing').freeze_support()
    main()
