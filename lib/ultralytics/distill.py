import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.yolo.detect.distill import DetectionDistiller

if __name__ == '__main__':
    param_dict = {
        # origin
        'model': '/your_path/ultralytics/cfg/models/mamba-yolo/Student.yaml',
        'data':'/your_path/dataset/data_plus.yaml',
        'imgsz': 640,
        'epochs': 300,
        'batch': 4,
        'workers': 8,
        'cache': True,
        'optimizer': 'SGD',
        'device': '1',
        'close_mosaic': 20,
        # 'amp': False, 
        'project':'/your_path/results',
        'name':'your_name',
        
        # distill
        'prune_model': False,
        'teacher_weights': '/your/Teacher.pt',
        'teacher_cfg': '/your_path/ultralytics/cfg/models/mamba-yolo/Teacher.yaml',
        'kd_loss_type': 'all',
        'kd_loss_decay': 'constant',
        
        'logical_loss_type': 'BCKD',
        'logical_loss_ratio': 1.0,
        
        'teacher_kd_layers': '14, 17, 20',
        'student_kd_layers': '14, 17, 20',
        'feature_loss_type': 'cwd',
        'feature_loss_ratio': 1.0
    }
    
    model = DetectionDistiller(overrides=param_dict)
    # model = SegmentationDistiller(overrides=param_dict)
    # model = PoseDistiller(overrides=param_dict)
    # model = OBBDistiller(overrides=param_dict)
    model.distill()
