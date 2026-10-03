"""Own-model retrained ablation: remove DINO branch and its fusion block.

This is not the complete ExCEL baseline: it retains our CE+Dice schedule and
does not introduce the original ExCEL LVC second-forward pathway.
"""
from model.model_excel import ExCEL_model
from research.protocol import model_options

class ResearchModel(ExCEL_model):
    def __init__(self,cfg):
        super().__init__(clip_model='ExCEL_ViT-B/16',embedding_dim=256,in_channels=768,
            img_size=cfg['crop'],mode='train',device='cuda',dino_model=None,
            **model_options(cfg))
        self.encoder.requires_grad_(False).eval()
    def train(self,mode=True):
        super().train(mode);self.encoder.eval();return self
    def forward(self,img):
        seg,fts,cams,attention,affinity=super().forward(img)
        return {'seg':seg,'features':fts,'cams':cams.detach(),'attention':attention.detach(),'affinity':affinity,'dense':None}
