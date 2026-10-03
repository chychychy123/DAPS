import torch
import torch.nn as nn
import torch.nn.functional as F
from .segformer_head import SegFormerHead
import numpy as np
import os
from torchvision.transforms import Compose, Normalize
from .decoder.TransDecoder import DecoderTransformer
from .decoder.TransDecoder_clip_dino import DecoderTransformer as ClipDinoDecoderTransformer  # 添加clip_dino解码器
import clip
from datasets.clip_text import new_class_names, BACKGROUND_CATEGORY,new_class_names_coco, BACKGROUND_CATEGORY_COCO
from .load_attr import attr_aggregate


class ExCEL_model(nn.Module):
    def __init__(self,  clip_model=None, embedding_dim=256, in_channels=512, dataset_name='pascal_voc', \
                        num_classes=21, num_atrr_clusters=112, json_file='./gpt4.0_cluster_a_photo_of4.json',\
                        img_size=320, mode='train', device='cuda', dino_model="dinov2_vitb14", dino_fts_dim=768, \
                        use_clip_dino_decoder=False):  
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.use_clip_dino_decoder = use_clip_dino_decoder  # 保存参数

        self.encoder, _ = clip.load(clip_model, device=device)
        self.encoder.visual.reload_self_attn(layers=6, feat_size=img_size//16, mode=mode)
        self.encoder.eval()
        self.in_channels = in_channels

         # 添加DINO模型支持
        if dino_model is not None:
            # 检查本地预训练模型是否存在
            local_dino_path = os.path.join(os.path.dirname(__file__), '..', 'pretrained', 'dinov2_vitb14_pretrain.pth')
            if os.path.exists(local_dino_path):
                print(f"从本地路径加载DINO模型权重: {local_dino_path}")
                
                # 直接创建ViT模型架构（不需要dinov2库）
                from transformers import ViTModel, ViTConfig
                
                # 根据dino_model参数创建对应的ViT配置
                if "vit_s" in dino_model.lower() or "s14" in dino_model:
                    config = ViTConfig(
                        hidden_size=384,
                        num_hidden_layers=12,
                        num_attention_heads=6,
                        intermediate_size=1536,
                        patch_size=14,
                        image_size=518  # DINOv2常用尺寸，确保位置编码数量匹配
                    )
                elif "vit_b" in dino_model.lower() or "b14" in dino_model:
                    config = ViTConfig(
                        hidden_size=768,
                        num_hidden_layers=12,
                        num_attention_heads=12,
                        intermediate_size=3072,
                        patch_size=14,
                        image_size=518  # DINOv2常用尺寸
                    )
                else:
                    # 默认使用base配置
                    config = ViTConfig(
                        hidden_size=768,
                        num_hidden_layers=12,
                        num_attention_heads=12,
                        intermediate_size=3072,
                        patch_size=14,
                        image_size=518  # DINOv2 vit_base使用518x518输入
                    )
                    
                self.dino_encoder = ViTModel(config)
                
                # 加载预训练权重
                state_dict = torch.load(local_dino_path)
                # 适配权重名称（HuggingFace和DINOv2权重命名差异）
                adapted_state_dict = {}
                for key, value in state_dict.items():
                    # 移除可能的前缀
                    new_key = key
                    if key.startswith('module.'):
                        new_key = key[7:]
                    
                    # 过滤掉register tokens相关的权重（dinov2_vitb14没有register tokens）
                    if 'register' in new_key.lower() or 'reg_token' in new_key.lower():
                        continue
                    
                    # 映射DINOv2权重名称到HuggingFace ViT名称
                    if new_key == 'cls_token':
                        new_key = 'embeddings.cls_token'
                    elif new_key == 'pos_embed':
                        new_key = 'embeddings.position_embeddings'
                    elif new_key == 'patch_embed.proj.weight':
                        new_key = 'embeddings.patch_embeddings.projection.weight'
                    elif new_key == 'patch_embed.proj.bias':
                        new_key = 'embeddings.patch_embeddings.projection.bias'
                    elif new_key.startswith('blocks.'):
                        # 映射blocks结构
                        parts = new_key.split('.')
                        layer_idx = parts[1]
                        if parts[2] == 'norm1':
                            if parts[3] == 'weight':
                                new_key = f'encoder.layer.{layer_idx}.layernorm_before.weight'
                            elif parts[3] == 'bias':
                                new_key = f'encoder.layer.{layer_idx}.layernorm_before.bias'
                        elif parts[2] == 'attn':
                            if parts[3] == 'qkv':
                                if parts[4] == 'weight':
                                    new_key = f'encoder.layer.{layer_idx}.attention.attention.query.weight'
                                    # 注意：DINO中的qkv是一个矩阵，需要拆分
                                    q_weight, k_weight, v_weight = value.chunk(3, dim=0)
                                    adapted_state_dict[new_key] = q_weight
                                    new_key = f'encoder.layer.{layer_idx}.attention.attention.key.weight'
                                    adapted_state_dict[new_key] = k_weight
                                    new_key = f'encoder.layer.{layer_idx}.attention.attention.value.weight'
                                    adapted_state_dict[new_key] = v_weight
                                    continue  # 跳过默认处理
                                elif parts[4] == 'bias':
                                    new_key = f'encoder.layer.{layer_idx}.attention.attention.query.bias'
                                    q_bias, k_bias, v_bias = value.chunk(3, dim=0)
                                    adapted_state_dict[new_key] = q_bias
                                    new_key = f'encoder.layer.{layer_idx}.attention.attention.key.bias'
                                    adapted_state_dict[new_key] = k_bias
                                    new_key = f'encoder.layer.{layer_idx}.attention.attention.value.bias'
                                    adapted_state_dict[new_key] = v_bias
                                    continue  # 跳过默认处理
                            elif parts[3] == 'proj':
                                if parts[4] == 'weight':
                                    new_key = f'encoder.layer.{layer_idx}.attention.output.dense.weight'
                                elif parts[4] == 'bias':
                                    new_key = f'encoder.layer.{layer_idx}.attention.output.dense.bias'
                        elif parts[2] == 'norm2':
                            if parts[3] == 'weight':
                                new_key = f'encoder.layer.{layer_idx}.layernorm_after.weight'
                            elif parts[3] == 'bias':
                                new_key = f'encoder.layer.{layer_idx}.layernorm_after.bias'
                        elif parts[2] == 'mlp':
                            if parts[3] == 'fc1':
                                if parts[4] == 'weight':
                                    new_key = f'encoder.layer.{layer_idx}.intermediate.dense.weight'
                                elif parts[4] == 'bias':
                                    new_key = f'encoder.layer.{layer_idx}.intermediate.dense.bias'
                            elif parts[3] == 'fc2':
                                if parts[4] == 'weight':
                                    new_key = f'encoder.layer.{layer_idx}.output.dense.weight'
                                elif parts[4] == 'bias':
                                    new_key = f'encoder.layer.{layer_idx}.output.dense.bias'
                    elif new_key == 'norm.weight':
                        new_key = 'layernorm.weight'
                    elif new_key == 'norm.bias':
                        new_key = 'layernorm.bias'
                    
                    adapted_state_dict[new_key] = value
                
                # 使用strict=False忽略不匹配的键
                self.dino_encoder.load_state_dict(adapted_state_dict, strict=False)
                print("成功加载本地DINO模型权重")
            else:
                raise FileNotFoundError(f"本地DINO模型文件不存在: {local_dino_path}")
            
            for name, param in self.dino_encoder.named_parameters():
                param.requires_grad = False
            
            self.dino_fts_fuse_dim = dino_fts_dim
            self.dino_decoder_fts_fuse = SegFormerHead(
                in_channels=self.dino_fts_fuse_dim, 
                embedding_dim=self.embedding_dim,
                num_classes=self.num_classes, 
                index=1
            )
            # 更新in_channels以适应CLIP和DINO特征连接后的维度
            fused_in_channels = self.embedding_dim*2
            self.fused_decoder_fts_fuse = SegFormerHead(in_channels=fused_in_channels, embedding_dim=self.embedding_dim,
                                              num_classes=self.num_classes, index=12)
        else:
            self.dino_encoder = None

        self.decoder_fts_fuse = SegFormerHead(in_channels=self.in_channels,embedding_dim=self.embedding_dim,
                                              num_classes=self.num_classes, index=12)
        
        # 根据参数选择使用哪种解码器
        if use_clip_dino_decoder and dino_model is not None:
            self.decoder = ClipDinoDecoderTransformer(width=self.embedding_dim, layers=3, heads=8, output_dim=self.num_classes)
        else:
            self.decoder = DecoderTransformer(width=self.embedding_dim, layers=3, heads=8, output_dim=self.num_classes)

        text_prompts = new_class_names+BACKGROUND_CATEGORY if num_classes <= 21 else new_class_names_coco+BACKGROUND_CATEGORY_COCO
        self.integral_text_features = clip.encode_text_with_prompt_ensemble(self.encoder, text_prompts, device, prompt_templates=['a clean origami {}.'])
        self.text_attr, self.attr_flag = attr_aggregate(self.integral_text_features, dataset_name, num_classes-1, num_atrr_clusters, json_file)

    def get_param_groups(self):

        param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)
        for param in list(self.decoder_fts_fuse.parameters()):
            param_groups[3].append(param)
        
        # 添加DINO相关参数组
        if hasattr(self, 'dino_decoder_fts_fuse'):
            for param in list(self.dino_decoder_fts_fuse.parameters()):
                param_groups[3].append(param)
        if hasattr(self, 'fused_decoder_fts_fuse'):
            for param in list(self.fused_decoder_fts_fuse.parameters()):
                param_groups[3].append(param)

        return param_groups

    def forward(self, img, ex_feats=None):
        if ex_feats is not None:
            image_features_, attn_weights_, all_feats_ = clip.generate_clip_fts(img, self.encoder, return_weights=True, ex_feats=ex_feats)
            attr_maps_raw_ = clip.clip_feature_surgery(image_features_, self.text_attr.permute(1,0))[:,1:,:self.num_classes-1]
            return attr_maps_raw_

        b, c, h, w = img.shape
        self.encoder.eval()
        image_features, attn_weights, all_feats = clip.generate_clip_fts(img, self.encoder, return_weights=True)
        attr_maps_raw = clip.clip_feature_surgery(image_features, self.text_attr.permute(1,0))[:,1:,:self.num_classes-1]
        # attr_maps_raw = clip.clip_feature_surgery(image_features, self.integral_text_features)[:,1:,:self.num_classes-1]

        all_img_tokens =  all_feats[:, :, 1:, ...]
        all_img_tokens = all_img_tokens.permute(0, 1, 3, 2)
        all_img_tokens = all_img_tokens.reshape(12, b, all_img_tokens.size(-2), h//16, w //16) #(11, b, c, h, w)

        # 处理CLIP特征
        clip_fts = self.decoder_fts_fuse(all_img_tokens)
        
        # 处理DINO特征和融合
        if self.dino_encoder is not None:
            with torch.no_grad():
                # DINOv2使用14x14的补丁大小
                dino_img_h, dino_img_w = (h//14)*14, (w//14)*14
                dino_img = F.interpolate(img, size=(dino_img_h, dino_img_w), mode='bilinear', align_corners=False)
                 # 使用HuggingFace ViTModel的正确方法，并启用位置编码插值
                dino_outputs = self.dino_encoder(dino_img, output_hidden_states=True, interpolate_pos_encoding=True)
                # 获取patch tokens (移除CLS token)
                dino_fts = dino_outputs.last_hidden_state[:, 1:]
                
            dino_fts = dino_fts.reshape([b, dino_img_h//14, dino_img_w//14, -1]).permute(0,3,1,2)
            _, _, dino_h, dino_w = dino_fts.shape
            dino_fts_processed = self.dino_decoder_fts_fuse(dino_fts.unsqueeze(0))
            
            # 调整DINO特征尺寸以匹配CLIP特征
            _, _, fts_h, fts_w = clip_fts.shape
            dino_fts_processed = F.interpolate(dino_fts_processed, size=(fts_h, fts_w), mode='bilinear', align_corners=False)
            
            # 融合CLIP和DINO特征
            fused_fts = torch.cat([clip_fts, dino_fts_processed], dim=1)
            # 使用融合特征的decoder
            fts = self.fused_decoder_fts_fuse(fused_fts.unsqueeze(0).repeat(12, 1, 1, 1, 1))
        else:
            fts = clip_fts

        attn_fts = fts.clone()
        _, _, fts_h, fts_w = fts.shape
        
        seg, seg_attn_weight_list = self.decoder(fts)
        
        f_b, f_c, f_h, f_w = attn_fts.shape
        attn_fts_flatten = attn_fts.reshape(f_b, f_c, f_h*f_w)
        attn_fts_flatten = F.normalize(attn_fts_flatten, dim=1)
        attn_pred = attn_fts_flatten.transpose(2, 1).bmm(attn_fts_flatten)
        attn_pred = (attn_pred - torch.mean(attn_pred) * 1.) * 3.0
        attn_pred = torch.sigmoid(attn_pred)

        return seg, attn_fts.clone().detach(), attr_maps_raw, attn_weights, attn_pred