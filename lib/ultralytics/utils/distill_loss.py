import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.checks import check_version
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors

import functools


class LogicalLoss(nn.Module):
    def __init__(self, hyp, model, distiller='l2', task='detect') -> None:
        super().__init__()
        
        if distiller in ['l2', 'l1']:
            if task == 'detect':
                self.logical_loss = OutputLoss(hyp, distiller)
            elif task == 'pose':
                self.logical_loss = OutputLoss_Pose(hyp, distiller)
                self.logical_loss.kpt_shape = model.kpt_shape
            elif task == 'obb':
                self.logical_loss = OutputLoss_OBB(hyp, distiller)
            elif task == 'segment':
                self.logical_loss = OutputLoss_Seg(hyp, distiller)
        elif distiller == 'BCKD':
             self.logical_loss = BCKD(hyp)
        
        self.logical_loss.nc = model.nc
        self.logical_loss.no = model.model[-1].no
        self.logical_loss.reg_max = model.model[-1].reg_max
        self.logical_loss.stride = model.model[-1].stride
        self.logical_loss.use_dfl = self.logical_loss.reg_max > 1
        self.logical_loss.device = next(model.parameters()).device
        self.logical_loss.proj = torch.arange(model.model[-1].reg_max, dtype=torch.float, device=self.logical_loss.device)
        
        if distiller == 'BCKD':
            self.logical_loss.assigner = TaskAlignedAssigner(topk=10, num_classes=model.nc, alpha=0.5, beta=6.0)
            self.logical_loss.assigner_o2o = TaskAlignedAssigner(topk=1, num_classes=model.nc, alpha=0.5, beta=6.0)
        
        self.distiller = distiller
            
    def forward(self, s_p, t_p, batch, o2o=False):
        assert len(s_p) == len(t_p)
        if self.distiller == 'BCKD':
            loss = self.logical_loss(s_p, t_p, batch, o2o=o2o)
        else:
            loss = self.logical_loss(s_p, t_p, batch)
        return loss


class OutputLoss(nn.Module):
    def __init__(self, hyp, distiller='l2'):
        super().__init__()
        
        if distiller == 'l2':
            box_loss = torch.nn.MSELoss(reduction='none')
            cls_loss = torch.nn.MSELoss(reduction='none')
        elif distiller == 'l1':
            box_loss = torch.nn.L1Loss(reduction='none')
            cls_loss = torch.nn.L1Loss(reduction='none')
        else:
            raise NotImplementedError
        
        self.box_loss = box_loss
        self.cls_loss = cls_loss
        self.hyp = hyp
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    def forward(self, s_p, t_p, batch):
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        
        s_pred_distri, s_pred_scores = torch.cat([xi.view(s_p[0].shape[0], self.no, -1) for xi in s_p], 2).split(
            (self.reg_max * 4, self.nc), 1)
        t_pred_distri, t_pred_scores = torch.cat([xi.view(t_p[0].shape[0], self.no, -1) for xi in t_p], 2).split(
            (self.reg_max * 4, self.nc), 1)
        
        t_obj_scale = t_pred_scores.sigmoid().max(1)[0].unsqueeze(1)
        
        lbox = torch.mean(self.box_loss(s_pred_distri, t_pred_distri) * t_obj_scale.repeat(1, self.reg_max * 4, 1))
        lcls = torch.mean(self.cls_loss(s_pred_scores, t_pred_scores) * t_obj_scale.repeat(1, self.nc, 1))
        
        lbox *= self.hyp.dfl
        lcls *= self.hyp.cls
        
        return (lbox + lcls) * s_p[0].size(0)

class OutputLoss_Pose(OutputLoss):
    def __init__(self, hyp, distiller='l2'):
        super().__init__(hyp, distiller)
    
    def forward(self, s_p, t_p, batch):
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lpose = torch.zeros(1, device=self.device)  # pose loss
        
        s_pred_distri, s_pred_scores = torch.cat([xi.view(s_p[0][0].shape[0], self.no, -1) for xi in s_p[0]], 2).split(
            (self.reg_max * 4, self.nc), 1)
        t_pred_distri, t_pred_scores = torch.cat([xi.view(t_p[0][0].shape[0], self.no, -1) for xi in t_p[0]], 2).split(
            (self.reg_max * 4, self.nc), 1)
        
        t_obj_scale = t_pred_scores.sigmoid().max(1)[0].unsqueeze(1)
        
        lbox = torch.mean(self.box_loss(s_pred_distri, t_pred_distri) * t_obj_scale.repeat(1, self.reg_max * 4, 1))
        lcls = torch.mean(self.cls_loss(s_pred_scores, t_pred_scores) * t_obj_scale.repeat(1, self.nc, 1))
        lpose = torch.mean(self.cls_loss(s_p[1], t_p[1]) * t_obj_scale.repeat(1, self.kpt_shape[0] * self.kpt_shape[1], 1))
        
        lbox *= self.hyp.dfl
        lcls *= self.hyp.cls
        lpose *= self.hyp.pose
        
        return (lbox + lcls + lpose) * s_p[0][0].size(0)

class OutputLoss_OBB(OutputLoss):
    def __init__(self, hyp, distiller='l2'):
        super().__init__(hyp, distiller)
    
    def forward(self, s_p, t_p, batch):
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobb = torch.zeros(1, device=self.device)  # obb loss
        
        s_pred_distri, s_pred_scores = torch.cat([xi.view(s_p[0][0].shape[0], self.no, -1) for xi in s_p[0]], 2).split(
            (self.reg_max * 4, self.nc), 1)
        t_pred_distri, t_pred_scores = torch.cat([xi.view(t_p[0][0].shape[0], self.no, -1) for xi in t_p[0]], 2).split(
            (self.reg_max * 4, self.nc), 1)
        
        t_obj_scale = t_pred_scores.sigmoid().max(1)[0].unsqueeze(1)
        
        lbox = torch.mean(self.box_loss(s_pred_distri, t_pred_distri) * t_obj_scale.repeat(1, self.reg_max * 4, 1))
        lcls = torch.mean(self.cls_loss(s_pred_scores, t_pred_scores) * t_obj_scale.repeat(1, self.nc, 1))
        lobb = torch.mean(self.cls_loss(s_p[1], t_p[1]) * t_obj_scale)
        
        lbox *= self.hyp.dfl
        lcls *= self.hyp.cls
        lobb *= self.hyp.box
        
        return (lbox + lcls + lobb) * s_p[0][0].size(0)

class OutputLoss_Seg(OutputLoss):
    def __init__(self, hyp, distiller='l2'):
        super().__init__(hyp, distiller)
    
    def forward(self, s_p, t_p, batch):
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lseg = torch.zeros(1, device=self.device)  # seg loss
        lproto = torch.zeros(1, device=self.device) # proto loss
        
        s_pred_distri, s_pred_scores = torch.cat([xi.view(s_p[0][0].shape[0], self.no, -1) for xi in s_p[0]], 2).split(
            (self.reg_max * 4, self.nc), 1)
        t_pred_distri, t_pred_scores = torch.cat([xi.view(t_p[0][0].shape[0], self.no, -1) for xi in t_p[0]], 2).split(
            (self.reg_max * 4, self.nc), 1)
        
        t_obj_scale = t_pred_scores.sigmoid().max(1)[0].unsqueeze(1)
        
        lbox = torch.mean(self.box_loss(s_pred_distri, t_pred_distri) * t_obj_scale.repeat(1, self.reg_max * 4, 1))
        lcls = torch.mean(self.cls_loss(s_pred_scores, t_pred_scores) * t_obj_scale.repeat(1, self.nc, 1))
        lseg = torch.mean(self.cls_loss(s_p[1], t_p[1]) * t_obj_scale.repeat(1, s_p[1].size(1), 1))
        lproto = torch.mean(self.cls_loss(s_p[2], t_p[2]))
        
        lbox *= self.hyp.dfl
        lcls *= self.hyp.cls
        lseg *= self.hyp.box
        lproto *= self.hyp.box
        
        return (lbox + lcls + lseg + lproto) * s_p[0][0].size(0)


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            # i.e., all labels of an image belong to ignore index.
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper

@weighted_loss
def knowledge_distillation_kl_div_loss(pred,
                                       soft_label,
                                       T,
                                       detach_target=True):
    r"""Loss function for knowledge distilling using KL divergence.

    Args:
        pred (Tensor): Predicted logits with shape (N, n + 1).
        soft_label (Tensor): Target logits with shape (N, N + 1).
        T (int): Temperature for distillation.
        detach_target (bool): Remove soft_label from automatic differentiation

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert pred.size() == soft_label.size()
    target = F.softmax(soft_label / T, dim=1)
    if detach_target:
        target = target.detach()

    kd_loss = F.kl_div(
        F.log_softmax(pred / T, dim=1), target, reduction='none').mean(1) * (
                      T * T)

    return kd_loss

class KnowledgeDistillationKLDivLoss(nn.Module):
    """Loss function for knowledge distilling using KL divergence.

    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
        T (int): Temperature for distillation.
    """

    def __init__(self, reduction='mean', loss_weight=1.0, T=10):
        super(KnowledgeDistillationKLDivLoss, self).__init__()
        assert T >= 1
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.T = T

    def forward(self,
                pred,
                soft_label,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (Tensor): Predicted logits with shape (N, n + 1).
            soft_label (Tensor): Target logits with shape (N, N + 1).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')

        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss_kd = self.loss_weight * knowledge_distillation_kl_div_loss(
            pred,
            soft_label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            T=self.T)

        return loss_kd

@weighted_loss
def novel_kd_loss(pred,
                  soft_label,
                  detach_target=True,
                  beta=1.0):
    r"""Loss function for knowledge distilling using KL divergence.

    Args:
        pred (Tensor): Predicted logits with shape (N, n + 1).
        soft_label (Tensor): Target logits with shape (N, N + 1).
        T (int): Temperature for distillation.
        detach_target (bool): Remove soft_label from automatic differentiation

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert pred.size() == soft_label.size()
    target = soft_label.sigmoid()
    score = pred.sigmoid()

    if detach_target:
        target = target.detach()

    scale_factor = target - score
    kd_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * scale_factor.abs().pow(beta)
    kd_loss = kd_loss.sum(dim=-1, keepdim=False)
    return kd_loss

class NovelKDLoss(nn.Module):
    """Loss function for knowledge distilling using KL divergence.

    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
        T (int): Temperature for distillation.
    """

    def __init__(self, reduction='mean', loss_weight=1.0, T=10, threshold=0.05):
        super(NovelKDLoss, self).__init__()
        # assert T >= 1
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.T = T
        self.threshold = threshold

    def forward(self,
                pred,
                soft_label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                beta=1.0):
        """Forward function.

        Args:
            pred (Tensor): Predicted logits with shape (N, n + 1).
            soft_label (Tensor): Target logits with shape (N, N + 1).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')

        reduction = (reduction_override if reduction_override else self.reduction)

        loss_kd = self.loss_weight * novel_kd_loss(pred, soft_label, weight, reduction=reduction, avg_factor=avg_factor, beta=beta)

        return loss_kd

class BCKD(nn.Module):
    def __init__(self, hyp) -> None:
        super().__init__()
        
        self.hyp = hyp
        
        self.loss_score_kd_ratio = 7.5
        self.loss_kd = NovelKDLoss(loss_weight=1.0)
        self.loss_ld = KnowledgeDistillationKLDivLoss(loss_weight=0.0, T=10)
    
    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)
    
    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out
    
    def forward(self, s_p, t_p, batch, o2o=False):
        s_pred_distri, s_pred_scores = torch.cat([xi.view(s_p[0].shape[0], self.no, -1) for xi in s_p], 2).split(
            (self.reg_max * 4, self.nc), 1)
        t_pred_distri, t_pred_scores = torch.cat([xi.view(t_p[0].shape[0], self.no, -1) for xi in t_p], 2).split(
            (self.reg_max * 4, self.nc), 1)

        s_pred_distri, s_pred_scores = s_pred_distri.permute(0, 2, 1).contiguous(), s_pred_scores.permute(0, 2, 1).contiguous()
        t_pred_distri, t_pred_scores = t_pred_distri.permute(0, 2, 1).contiguous(), t_pred_scores.permute(0, 2, 1).contiguous()
        
        dtype = s_pred_scores.dtype
        batch_size = s_pred_scores.shape[0]
        imgsz = torch.tensor(s_p[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(s_p, self.stride, 0.5)
        
        # targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        
        s_pred_bboxes, t_pred_bboxes = self.bbox_decode(anchor_points, s_pred_distri), self.bbox_decode(anchor_points, t_pred_distri)
        
        if o2o:
            _, target_bboxes, target_scores, fg_mask, _ = self.assigner_o2o(
                s_pred_scores.detach().sigmoid(), (s_pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)  
        else:
            _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
                    s_pred_scores.detach().sigmoid(), (s_pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                    anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
        target_scores_sum = max(target_scores.sum(), 1)
        
        weight_targets1 = s_pred_scores.detach().sigmoid()
        weight_targets2 = t_pred_scores.detach().sigmoid()
        weight_targets3 = weight_targets2 - weight_targets1
        total_weight_targets = weight_targets3.abs().pow(1.0).max(dim=-1)[0] * fg_mask
        
        loss_score_kd = ((1 - bbox_iou(s_pred_bboxes, t_pred_bboxes.detach(), xywh=False, CIoU=True).squeeze()) * total_weight_targets).sum() / target_scores_sum
        loss_score_kd *= self.loss_score_kd_ratio
        
        fg_mask = fg_mask.bool()
        weight_targets = s_pred_scores.detach().sigmoid() + 1e-4
        weight_targets = weight_targets.max(dim=-1)[0][fg_mask] + 1e-4
        s_pred_corners, t_pred_corners = s_pred_distri[fg_mask].reshape(-1, self.reg_max), t_pred_distri[fg_mask].reshape(-1, self.reg_max)
        loss_ld = self.loss_ld(s_pred_corners, t_pred_corners, weight_targets[:, None].expand(-1, 4).reshape(-1), 4.0)
        loss_kd = self.loss_kd(s_pred_scores, t_pred_scores, fg_mask, target_scores_sum)
        
        return loss_score_kd + loss_ld + loss_kd

class FeatureLoss(nn.Module):
    def __init__(self,
                 channels_s,
                 channels_t,
                 distiller='cwd'):
        super(FeatureLoss, self).__init__()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.align_module = nn.ModuleList([
            nn.Conv2d(channel, tea_channel, kernel_size=1, stride=1,
                      padding=0).to(device) if channel != tea_channel else nn.Identity()
            for channel, tea_channel in zip(channels_s, channels_t)
        ])
        self.norm = [
            nn.BatchNorm2d(tea_channel, affine=False).to(device)
            for tea_channel in channels_t
        ]
        
        if (distiller == 'cwd'):
            self.feature_loss = CWDLoss(channels_s, channels_t)
        else:
            raise NotImplementedError

    def forward(self, y_s, y_t):
        assert len(y_s) == len(y_t)
        tea_feats = []
        stu_feats = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            s = self.align_module[idx](s)
            s = self.norm[idx](s)
            t = self.norm[idx](t)
            tea_feats.append(t)
            stu_feats.append(s)

        loss = self.feature_loss(stu_feats, tea_feats)
        return loss


class CWDLoss(nn.Module):

    def __init__(self, channels_s, channels_t, tau=1.0):
        super(CWDLoss, self).__init__()
        self.tau = tau

    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            N, C, H, W = s.shape
            # normalize in channel diemension
            softmax_pred_T = F.softmax(t.view(-1, W * H) / self.tau,
                                       dim=1)  # [N*C, H*W]

            logsoftmax = torch.nn.LogSoftmax(dim=1)
            cost = torch.sum(
                softmax_pred_T * logsoftmax(t.view(-1, W * H) / self.tau) -
                softmax_pred_T * logsoftmax(s.view(-1, W * H) / self.tau)) * (
                    self.tau**2)
            # cost = torch.sum(-softmax_pred_T * logsoftmax(s.view(-1, W * H)/self.tau)) * (self.tau ** 2)

            losses.append(cost / (C * N))
        loss = sum(losses)

        return loss

