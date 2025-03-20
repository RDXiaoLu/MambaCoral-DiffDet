import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO



if __name__ == '__main__':
    model = YOLO('/your_path/weights/best.pt')
    model.val(data='/mnt/hdd/yuanhui/ultralytics-main/dataset/data.yaml',
              split='val',
              imgsz=640,
              batch=32,
              # iou=0.5,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='your_path/results',
              name='your_project',
              )
