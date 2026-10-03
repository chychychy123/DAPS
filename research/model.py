import math
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
import clip
from model.model_excel import ExCEL_model
from research.protocol import model_options


class FrozenDino(nn.Module):
    """Official architecture and strict weights, never an approximate ViT mapping."""
    def __init__(self, repo, weights, name='dinov2_vitb14'):
        super().__init__()
        if not Path(weights).is_file():
            raise FileNotFoundError(weights)
        if name.startswith('dinov3'):
            self.net = torch.hub.load(repo, name, source='local', weights=weights)
        else:
            self.net = torch.hub.load(repo, name, source='local', pretrained=False)
            self.net.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
        self.net.requires_grad_(False).eval()
        self.dim = self.net.embed_dim
        patch = self.net.patch_size
        self.patch = patch[0] if isinstance(patch, (list, tuple)) else patch

    def train(self, mode=True):
        super().train(False)
        self.net.eval()
        return self

    @torch.no_grad()
    def forward(self, x):
        h, w = x.shape[-2:]
        dh, dw = math.ceil(h/self.patch)*self.patch, math.ceil(w/self.patch)*self.patch
        x = F.interpolate(x, (dh,dw), mode='bicubic', align_corners=False)
        tokens = self.net.forward_features(x)['x_norm_patchtokens']
        return tokens.transpose(1,2).reshape(x.shape[0], self.dim, dh//self.patch, dw//self.patch).float()


class ResearchModel(ExCEL_model):
    def __init__(self, config):
        super().__init__(clip_model='ExCEL_ViT-B/16', embedding_dim=256,
                         in_channels=768, img_size=config['crop'], mode='train',
                         **model_options(config),
                         dino_model=None)
        self.encoder.requires_grad_(False).eval()
        self.dense = FrozenDino(config['dino_repo'], config['dino_weights'], config['dino_name'])
        self.dense_proj = nn.Sequential(nn.Conv2d(self.dense.dim,256,1), nn.GELU(), nn.Conv2d(256,256,1))
        self.fuse = nn.Sequential(nn.Conv2d(512,256,1), nn.GELU(), nn.Dropout2d(.1))

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        self.dense.eval()
        return self

    def forward(self, img):
        b, _, h, w = img.shape
        with torch.no_grad():
            image_features, attention, features = clip.generate_clip_fts(img,self.encoder,return_weights=True)
            cams=clip.clip_feature_surgery(image_features,self.text_attr.permute(1,0))[:,1:,:self.num_classes-1]
            dense=self.dense(img)
        features=features[:,:,1:].permute(0,1,3,2).reshape(12,b,-1,h//16,w//16)
        clip_features=self.decoder_fts_fuse(features)
        spatial=F.interpolate(self.dense_proj(dense),clip_features.shape[-2:],mode='bilinear',align_corners=False)
        fused=self.fuse(torch.cat((clip_features,spatial),1))
        seg,_=self.decoder(fused)
        flat=F.normalize(fused.flatten(2),dim=1)
        affinity=flat.transpose(1,2).bmm(flat)
        affinity=torch.sigmoid((affinity-affinity.mean())*3)
        return {'seg':seg,'cams':cams.detach(),'attention':attention.detach(),
                'affinity':affinity,'dense':F.interpolate(dense,(h//16,w//16),mode='bilinear',align_corners=False)}
