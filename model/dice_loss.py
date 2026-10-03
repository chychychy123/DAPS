import torch
import torch.nn.functional as F

def dice_loss(output, target, eps=1e-7):
    """
    Dice Loss for semantic segmentation
    对小目标和类别不平衡更鲁棒
    
    Args:
        output: NCHW format, model prediction logits
        target: NHW format, ground truth labels
        eps: epsilon for numerical stability
    """
    eps = 1e-7
    num_classes = output.shape[1]
    
    # convert logits to probs (保持梯度)
    pred = F.softmax(output, dim=1)  # [B, C, H, W]
    
    # 处理ignore_index (255)
    target = target.clone()
    ignore_mask = (target == 255)
    target[ignore_mask] = 0  # 临时设置为0，后面会mask掉
    
    # convert target to onehot (保持在同一设备)
    # 使用F.one_hot而不是torch.eye，避免断开梯度
    target_one_hot = F.one_hot(target.long(), num_classes=num_classes)  # [B, H, W, C]
    target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]
    
    # 创建valid mask (忽略ignore_index的位置)
    valid_mask = (~ignore_mask).unsqueeze(1).float()  # [B, 1, H, W]
    valid_mask = valid_mask.expand_as(pred)  # [B, C, H, W]
    
    # 应用mask
    pred_masked = pred * valid_mask
    target_masked = target_one_hot * valid_mask
    
    # 计算每个类别的交集和并集
    inter = (pred_masked * target_masked).sum(dim=[2, 3])  # [B, C]
    union = (pred_masked + target_masked).sum(dim=[2, 3])  # [B, C]
    
    # 计算dice系数 (对所有batch和类别求平均)
    dice = (2. * inter + eps) / (union + eps)  # [B, C]
    dice = dice.mean()  # 全局平均
    
    return 1. - dice
    

class DiceLoss(torch.nn.Module):
    """
    Dice Loss Module
    用于替代或补充交叉熵损失
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, output, targ):
        """
        output is NCHW, targ is NHW
        """
        return dice_loss(output, targ)

    def activation(self, output):
        return F.softmax(output, dim=1)
    
    def decodes(self, output):
        return output.argmax(1)

