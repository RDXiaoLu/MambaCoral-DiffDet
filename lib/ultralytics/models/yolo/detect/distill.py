# Ultralytics YOLO 🚀, AGPL-3.0 license

import sys, os, torch, math, time, warnings
import matplotlib
matplotlib.use('AGG')
import matplotlib.pylab as plt
import torch.nn as nn
from torch import optim
from thop import clever_format
from functools import partial
from torch import distributed as dist
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime

from copy import copy, deepcopy
from pathlib import Path

import numpy as np

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight, attempt_load_weights
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, TQDM, clean_url, colorstr, emojis, yaml_save, callbacks, __version__
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first
from ultralytics.utils.checks import check_imgsz, print_args, check_amp
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.torch_utils import ModelEMA, EarlyStopping, one_cycle, init_seeds, select_device
from ultralytics.utils.distill_loss import LogicalLoss, FeatureLoss
from ultralytics.nn.extra_modules.kernel_warehouse import get_temperature

def get_activation(feat, backbone_idx=-1):
    def hook(model, inputs, outputs):
        if backbone_idx != -1:
            for _ in range(5 - len(outputs)): outputs.insert(0, None)
            # for idx, i in enumerate(outputs):
            #     if i is None:
            #         print(idx, 'None')
            #     else:
            #         print(idx, i.size())
            feat.append(outputs[backbone_idx])
        else:
            feat.append(outputs)
    return hook

class DetectionDistiller(BaseTrainer):
    """
    A class extending the BaseTrainer class for training based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionTrainer

        args = dict(model='yolov8n.pt', data='coco8.yaml', epochs=3)
        trainer = DetectionTrainer(overrides=args)
        trainer.train()
        ```
    """
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the BaseTrainer class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        self.check_resume(overrides)
        self.device = select_device(self.args.device, self.args.batch)
        self.validator = None
        self.model = None
        self.metrics = None
        self.plots = {}
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)

        # Dirs
        self.save_dir = get_save_dir(self.args)
        self.args.name = self.save_dir.name  # update name for loggers
        self.wdir = self.save_dir / 'weights'  # weights dir
        if RANK in (-1, 0):
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.args.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / 'args.yaml', vars(self.args))  # save run args
        self.last, self.best = self.wdir / 'last.pt', self.wdir / 'best.pt'  # checkpoint paths
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        # Device
        if self.device.type in ('cpu', 'mps'):
            self.args.workers = 0  # faster CPU training as time dominated by inference, not dataloading

        # Model and Dataset
        self.model = self.args.model
        try:
            if self.args.task == 'classify':
                self.data = check_cls_dataset(self.args.data)
            elif self.args.data.split('.')[-1] in ('yaml', 'yml') or self.args.task in ('detect', 'segment', 'pose'):
                self.data = check_det_dataset(self.args.data)
                if 'yaml_file' in self.data:
                    self.args.data = self.data['yaml_file']  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e

        self.trainset, self.testset = self.get_dataset()
        self.ema = None

        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.logical_disloss = None
        self.feature_disloss = None
        self.loss_names = ['Loss']
        self.csv = self.save_dir / 'results.csv'
        self.plot_idx = [0, 1, 2]

        # Callbacks
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        if RANK in (-1, 0):
            callbacks.add_integration_callbacks(self)
    
    def build_dataset(self, img_path, mode='train', batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == 'val', stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        """Construct and return dataloader."""
        assert mode in ['train', 'val']
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == 'train'
        if getattr(dataset, 'rect', False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == 'train' else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255
        return batch

    def set_model_attributes(self):
        """Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data['nc']  # attach number of classes to model
        self.model.names = self.data['names']  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = DetectionModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
        return yolo.detect.DetectionValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def label_loss_items(self, loss_items=None, prefix='train'):
        """
        Returns a loss dict with labelled training loss items tensor.

        Not needed for classification but necessary for segmentation & detection
        """
        keys = [f'{prefix}/{x}' for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ('\n' + '%11s' *
                (4 + len(self.loss_names) + 2)) % ('Epoch', 'GPU_mem', *self.loss_names, 'log_loss', 'fea_loss', 'Instances', 'Size')

    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        plot_images(images=batch['img'],
                    batch_idx=batch['batch_idx'],
                    cls=batch['cls'].squeeze(-1),
                    bboxes=batch['bboxes'],
                    paths=batch['im_file'],
                    fname=self.save_dir / f'train_batch{ni}.jpg',
                    on_plot=self.on_plot)

    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb['bboxes'] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb['cls'] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data['names'], save_dir=self.save_dir, on_plot=self.on_plot)

    def build_optimizer(self, model, name='auto', lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5, feature_model=None):
        """
        Constructs an optimizer for the given model, based on the specified optimizer name, learning rate, momentum,
        weight decay, and number of iterations.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations. Default: 'auto'.
            lr (float, optional): The learning rate for the optimizer. Default: 0.001.
            momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
            decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'. Default: 1e5.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """

        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
        if name == 'auto':
            LOGGER.info(f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                        f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                        f"determining best 'optimizer', 'lr0' and 'momentum' automatically... ")
            nc = getattr(model, 'nc', 10)  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ('SGD', 0.01, 0.9) if iterations > 10000 else ('AdamW', lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f'{module_name}.{param_name}' if module_name else param_name
                if 'bias' in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        if feature_model is not None:
            for v in feature_model.modules():
                for p_name, p in v.named_parameters(recurse=0):
                    if p_name == 'bias':  # bias (no decay)
                        g[2].append(p)
                    elif p_name == 'weight' and isinstance(v, bn):  # weight (no decay)
                        g[1].append(p)
                    else:
                        g[0].append(p)  # weight (with decay)

        if name in ('Adam', 'Adamax', 'AdamW', 'NAdam', 'RAdam'):
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == 'RMSProp':
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == 'SGD':
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers "
                f'[Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto].'
                'To request support for addition optimizers please visit https://github.com/ultralytics/ultralytics.')

        optimizer.add_param_group({'params': g[0], 'weight_decay': decay})  # add g0 with weight_decay
        optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)')
        return optimizer
    
    def setup_prune_model(self):
        ckpt = torch.load(self.model, map_location=self.device)
        model = ckpt['ema' if ckpt.get('ema') else 'model'].float()
        for p in model.parameters():
            p.requires_grad_(True)
        LOGGER.info(f"{colorstr('Loading Prune Student Model form {}'.format(self.model))}")
        self.model = model
        self.model.info()
        return ckpt
    
    def setup_model(self):
        """Load/create/download model for any task."""
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        model, weights = self.model, None
        ckpt = None
        if str(model).endswith('.pt'):
            weights, ckpt = attempt_load_one_weight(model)
            cfg = weights.yaml
        else:
            cfg = model
        self.model = self.get_model(cfg=cfg, weights=self.pretrain_weights, verbose=RANK == -1)  # calls Model(cfg, weights)
        return ckpt
    
    def setup_teacher_model(self):
        """Load/create/download model for any task."""
        model, weights = self.args.teacher_weights, None
        ckpt = None
        if str(model).endswith('.pt'):
            weights, ckpt = attempt_load_one_weight(model)
            cfg = weights.yaml
        else:
            cfg = model
        self.teacher_model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)
        return ckpt
    
    def _setup_train(self, world_size):
        # init model
        if self.args.prune_model:
            ckpt = self.setup_prune_model()
        else:
            LOGGER.info(f"{colorstr('SetUp Student Model:')}")
            ckpt = self.setup_model()
        LOGGER.info(f"{colorstr('SetUp Teacher Model:')}")
        _ = self.setup_teacher_model()
        
        self.model.to(self.device)
        self.teacher_model.to(self.device)
        self.set_model_attributes()
        self.model.criterion = self.model.init_criterion()
        
        # Freeze layers
        freeze_list = self.args.freeze if isinstance(
            self.args.freeze, list) else range(self.args.freeze) if isinstance(self.args.freeze, int) else []
        always_freeze_names = ['.dfl']  # always freeze these layers
        freeze_layer_names = [f'model.{x}.' for x in freeze_list] + always_freeze_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad:
                LOGGER.info(f"WARNING ⚠️ setting 'requires_grad=True' for frozen layer '{k}'. "
                            'See ultralytics.engine.trainer for customization of frozen layers.')
                v.requires_grad = True
        
        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in (-1, 0):  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and world_size > 1:  # DDP
            dist.broadcast(self.amp, src=0)  # broadcast the tensor from rank 0 to all other ranks (returns None)
        self.amp = bool(self.amp)  # as boolean
        self.scaler = amp.GradScaler(enabled=self.amp)
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[RANK])
        
        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, 'stride') else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)

        # Batch size
        if self.batch_size == -1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = check_train_batch_size(self.model, self.args.imgsz, self.amp)

        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=RANK, mode='train')
        if RANK in (-1, 0):
            self.test_loader = self.get_dataloader(self.testset, batch_size=batch_size * 2, rank=-1, mode='val')
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix='val')
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args.plots:
                self.plot_training_labels()

        # Init Distill Loss
        self.kd_logical_loss, self.kd_feature_loss = None, None
        if self.args.kd_loss_type == 'logical' or self.args.kd_loss_type == 'all':
            self.kd_logical_loss = LogicalLoss(self.args, self.model, self.args.logical_loss_type)
        if self.args.kd_loss_type == 'feature' or self.args.kd_loss_type == 'all':
            s_feature, t_feature = [], []
            hooks = []
            self.teacher_kd_layers, self.student_kd_layers = self.args.teacher_kd_layers.split(','), self.args.student_kd_layers.split(',')
            assert len(self.teacher_kd_layers) == len(self.student_kd_layers), f"teacher{self.teacher_kd_layers} and student{self.student_kd_layers} layers not equal.."
            for t_layer, s_layer in zip(self.teacher_kd_layers, self.student_kd_layers):
                if '-' in t_layer:
                    t_layer_first, t_layer_second = t_layer.split('-')
                    hooks.append(de_parallel(self.teacher_model).model[int(t_layer_first)].register_forward_hook(get_activation(t_feature, backbone_idx=int(t_layer_second))))
                else:
                    hooks.append(de_parallel(self.teacher_model).model[int(t_layer)].register_forward_hook(get_activation(t_feature)))
                
                if '-' in s_layer:
                    s_layer_first, s_layer_second = s_layer.split('-')
                    hooks.append(de_parallel(self.model).model[int(s_layer_first)].register_forward_hook(get_activation(s_feature, backbone_idx=int(s_layer_second))))
                else:
                    hooks.append(de_parallel(self.model).model[int(s_layer)].register_forward_hook(get_activation(s_feature)))
            inputs = torch.randn((2, 3, self.args.imgsz, self.args.imgsz)).to(self.device)
            with torch.no_grad():
                _ = self.teacher_model(inputs)
                _ = self.model(inputs)
            self.kd_feature_loss = FeatureLoss([i.size(1) for i in s_feature], [i.size(1) for i in t_feature], distiller=self.args.feature_loss_type)
            for hook in hooks:
                hook.remove()
        
        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(model=self.model,
                                              name=self.args.optimizer,
                                              lr=self.args.lr0,
                                              momentum=self.args.momentum,
                                              decay=weight_decay,
                                              iterations=iterations,
                                              feature_model=self.kd_feature_loss if isinstance(self.kd_feature_loss, FeatureLoss) else None)
        # Scheduler
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: (1 - x / self.epochs) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(None)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks('on_pretrain_routine_end')
    
    def _do_train(self, world_size=1):
        """Train completed, evaluate and plot if specified by arguments."""
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.run_callbacks('on_train_start')
        LOGGER.info(f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
                    f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
                    f"Logging results to {colorstr('bold', self.save_dir)}\n"
                    f'Starting training for {self.epochs} epochs...')
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.epochs  # predefine for resume fully trained model edge cases
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.run_callbacks('on_train_epoch_start')
            self.model.train()
            
            if self.args.kd_loss_type in ['feature', 'all']:
                self.kd_feature_loss.train()
                hooks = []
                s_feature, t_feature = [], []
                for t_layer, s_layer in zip(self.teacher_kd_layers, self.student_kd_layers):
                    if '-' in t_layer:
                        t_layer_first, t_layer_second = t_layer.split('-')
                        hooks.append(de_parallel(self.teacher_model).model[int(t_layer_first)].register_forward_hook(get_activation(t_feature, backbone_idx=int(t_layer_second))))
                    else:
                        hooks.append(de_parallel(self.teacher_model).model[int(t_layer)].register_forward_hook(get_activation(t_feature)))
                    
                    if '-' in s_layer:
                        s_layer_first, s_layer_second = s_layer.split('-')
                        hooks.append(de_parallel(self.model).model[int(s_layer_first)].register_forward_hook(get_activation(s_feature, backbone_idx=int(s_layer_second))))
                    else:
                        hooks.append(de_parallel(self.model).model[int(s_layer)].register_forward_hook(get_activation(s_feature)))
            
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                LOGGER.info('Closing dataloader mosaic')
                if hasattr(self.train_loader.dataset, 'mosaic'):
                    self.train_loader.dataset.mosaic = False
                if hasattr(self.train_loader.dataset, 'close_mosaic'):
                    self.train_loader.dataset.close_mosaic(hyp=self.args)
                self.train_loader.reset()

            if RANK in (-1, 0):
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            self.logical_disloss = torch.zeros(1, device=self.device)
            self.feature_disloss = torch.zeros(1, device=self.device)
            self.optimizer.zero_grad()
            for i, batch in pbar:
                self.run_callbacks('on_train_batch_start')
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])
                
                if hasattr(self.model, 'net_update_temperature'):
                    temp = get_temperature(i + 1, epoch, len(self.train_loader), temp_epoch=20, temp_init_value=1.0)
                    self.model.net_update_temperature(temp)
                
                if self.args.kd_loss_decay == 'constant':
                    distill_decay = 1.0
                elif self.args.kd_loss_decay == 'cosine':
                    eta_min, base_ratio, T_max = 0.01, 1.0, 10
                    distill_decay = eta_min + (base_ratio - eta_min) * (1 + math.cos(math.pi * i / T_max)) / 2
                elif self.args.kd_loss_decay == 'linear':
                    distill_decay = ((1 - math.cos(i * math.pi / len(self.train_loader))) / 2) * (0.01 - 1) + 1
                elif self.args.kd_loss_decay == 'cosine_epoch':
                    eta_min, base_ratio, T_max = 0.01, 1.0, 10
                    distill_decay = eta_min + (base_ratio - eta_min) * (1 + math.cos(math.pi * ni / T_max)) / 2
                elif self.args.kd_loss_decay == 'linear_epoch':
                    distill_decay = ((1 - math.cos(ni * math.pi / (self.epochs * nb))) / 2) * (0.01 - 1) + 1
                
                # Forward
                with torch.cuda.amp.autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    pred = self.model.predict(batch['img'])
                    
                    with torch.no_grad():
                        t_pred = self.teacher_model.predict(batch['img'])
                    
                    main_loss, self.loss_items = self.model.criterion(pred, batch)
                    
                    log_distill_loss, fea_distill_loss = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
                    if self.kd_logical_loss is not None:
                        if type(pred) is dict and type(t_pred) is dict:
                            log_distill_loss = self.kd_logical_loss(pred['one2one'], t_pred['one2one'], batch, True) * self.args.logical_loss_ratio
                            log_distill_loss += self.kd_logical_loss(pred['one2many'], t_pred['one2many'], batch) * self.args.logical_loss_ratio * 0.5
                        else:
                            log_distill_loss = self.kd_logical_loss(pred, t_pred, batch) * self.args.logical_loss_ratio
                    if self.kd_feature_loss is not None:
                        fea_distill_loss = self.kd_feature_loss(s_feature, t_feature) * self.args.feature_loss_ratio
                    
                    self.loss = main_loss + (log_distill_loss + fea_distill_loss) * batch['img'].size(0) * distill_decay
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None \
                        else self.loss_items
                    self.logical_disloss = (self.logical_disloss * i + log_distill_loss) / (i + 1) if self.logical_disloss is not None \
                        else log_distill_loss
                    self.feature_disloss = (self.feature_disloss * i + fea_distill_loss) / (i + 1) if self.feature_disloss is not None \
                        else fea_distill_loss

                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                # Log
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                logical_dislosses = self.logical_disloss if loss_len > 1 else torch.unsqueeze(self.logical_disloss, 0)
                feature_dislosses = self.feature_disloss if loss_len > 1 else torch.unsqueeze(self.feature_disloss, 0)
                if RANK in (-1, 0):
                    pbar.set_description(
                        ('%11s' * 2 + '%11.4g' * (2 + loss_len + 2)) %
                        (f'{epoch + 1}/{self.epochs}', mem, *losses, *logical_dislosses, *feature_dislosses, batch['cls'].shape[0], batch['img'].shape[-1]))
                    self.run_callbacks('on_batch_end')
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks('on_train_batch_end')
                
                if self.kd_feature_loss is not None:
                    s_feature.clear()
                    t_feature.clear()

            self.lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()
            self.run_callbacks('on_train_epoch_end')
            
            if self.kd_feature_loss is not None:
                for hook in hooks:
                    hook.remove()

            if RANK in (-1, 0):

                # Validation
                self.ema.update_attr(self.model, include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])
                final_epoch = (epoch + 1 == self.epochs) or self.stopper.possible_stop

                if self.args.val or final_epoch:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop = self.stopper(epoch + 1, self.fitness)

                # Save model
                if self.args.save or (epoch + 1 == self.epochs):
                    self.save_model()
                    self.run_callbacks('on_model_save')

            tnow = time.time()
            self.epoch_time = tnow - self.epoch_time_start
            self.epoch_time_start = tnow
            self.run_callbacks('on_fit_epoch_end')
            torch.cuda.empty_cache()  # clears GPU vRAM at end of epoch, can help with out of memory errors

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                if RANK != 0:
                    self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks

        if RANK in (-1, 0):
            # Do final val with best.pt
            LOGGER.info(f'\n{epoch - self.start_epoch + 1} epochs completed in '
                        f'{(time.time() - self.train_time_start) / 3600:.3f} hours.')
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks('on_train_end')
        torch.cuda.empty_cache()
        self.run_callbacks('teardown')
        
    def distill(self, weights=None):
        self.pretrain_weights = weights
        self.train()