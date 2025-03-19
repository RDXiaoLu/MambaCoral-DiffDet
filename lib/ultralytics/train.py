import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO



if __name__ == '__main__':
    model = YOLO('/your_path/ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml')
    # model.load('mamba_yolo_t.pt') # loading pretrain weights
    model.train(data='/your_path/dataset/data_plus.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=32,
                close_mosaic=0,
                workers=4,
                # device='0',
                optimizer='SGD', # using SGD
                # patience=0, # close earlystop
                # resume=True, # Breakpoint training, select last.pt when YOLO is initialized
                # amp=False, # close amp
                # fraction=0.2,
                project='./results',
                name='Mamba_YOLO_T',
                device=[0]
                )
