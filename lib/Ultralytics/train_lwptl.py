import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# 训练参数官方详解链接：https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings:~:text=a%20training%20run.-,Train%20Settings,-The%20training%20settings

if __name__ == '__main__':
    model.load('MCDD.pt') # loading pretrain weights
    model.train(data='/your_path/dataset/data_lwptl.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=4,
                close_mosaic=0,
                workers=4,
                # device='0',
                optimizer='SGD', # using SGD
                # patience=0, # close earlystop
                # resume=True, # 断点续训,YOLO初始化时选择last.pt
                # amp=False, # close amp
                # fraction=0.2,
                project='./results',
                name='MCDD_lwptl',
                device=[0]
                )
