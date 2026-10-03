import torch
import torch.nn as nn
from .dice_loss import DiceLoss

# 初始化Dice Loss
dice_loss_fn = DiceLoss()

def get_seg_loss(pred, label, ignore_index=255, use_dice=True, dice_weight=0.3):
    """
    改进的分割损失函数，结合交叉熵和Dice Loss
    
    Args:
        pred: 预测logits
        label: 真实标签
        ignore_index: 忽略的索引
        use_dice: 是否使用Dice Loss (from WeCLIP+)
        dice_weight: Dice Loss的权重
    """
    ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    bg_label = label.clone()
    bg_label[label!=0] = ignore_index
    bg_sum = (bg_label != ignore_index).long().sum()
    bg_loss = ce(pred,bg_label.type(torch.long)).sum()/(bg_sum + 1e-6)
    
    fg_label = label.clone()
    fg_label[label==0] = ignore_index
    fg_sum = (fg_label != ignore_index).long().sum()
    fg_loss = ce(pred,fg_label.type(torch.long)).sum()/(fg_sum + 1e-6)

    ce_loss = (bg_loss + fg_loss) * 0.5
    
    # 添加Dice Loss以改善小目标和类别不平衡问题
    if use_dice:
        d_loss = dice_loss_fn(pred, label)
        total_loss = (1 - dice_weight) * ce_loss + dice_weight * d_loss
        return total_loss
    
    return ce_loss

def get_aff_loss(inputs, targets):

    pos_label = (targets == 1).type(torch.int16)
    pos_count = pos_label.sum() + 1
    neg_label = (targets == 0).type(torch.int16)
    neg_count = neg_label.sum() + 1
    #inputs = torch.sigmoid(input=inputs)

    pos_loss = torch.sum(pos_label * (1 - inputs)) / pos_count
    neg_loss = torch.sum(neg_label * (inputs)) / neg_count

    return 0.5 * pos_loss + 0.5 * neg_loss, pos_count, neg_count


def get_dual_modal_loss(seg_clip, seg_dino, label, ignore_index=255, consistency_weight=0.2):
    """
    双模态互补监督损失 (from WeCLIP+)
    让CLIP和DINO的预测互相学习，提高模型鲁棒性
    
    Args:
        seg_clip: CLIP分支的预测
        seg_dino: DINO分支的预测  
        label: 真实标签
        consistency_weight: 一致性损失权重
    """
    # 各自的监督损失
    loss_clip = get_seg_loss(seg_clip, label, ignore_index, use_dice=True)
    loss_dino = get_seg_loss(seg_dino, label, ignore_index, use_dice=True)
    
    # 互补监督：让两个分支的预测保持一致
    # 使用KL散度衡量两个分布的差异
    pred_clip = torch.softmax(seg_clip, dim=1)
    pred_dino = torch.softmax(seg_dino, dim=1)
    
    # 双向KL散度
    consistency_loss = (
        torch.nn.functional.kl_div(pred_clip.log(), pred_dino.detach(), reduction='batchmean') +
        torch.nn.functional.kl_div(pred_dino.log(), pred_clip.detach(), reduction='batchmean')
    ) * 0.5
    
    total_loss = loss_clip + loss_dino + consistency_weight * consistency_loss
    
    return total_loss, loss_clip, loss_dino, consistency_loss