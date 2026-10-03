"""Experimental hypotheses, not claimed novel or effective before ablation."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def anchored_graph(cams, dense, labels, topk=16, steps=3, strength=.5):
    """Mutual-neighbour diffusion with fixed reliable seeds and class veto.

    cams: B,C,H,W, already normalized to [0,1]; dense: B,D,H,W.
    Outputs foreground maps plus an independent DINO-prototype distribution.
    """
    b,c,h,w=cams.shape
    scores=cams.float().clamp(0,1)*labels[:,:,None,None]
    q=torch.cat(((1-scores.max(1,keepdim=True).values).clamp(0,1),scores),1)
    q=q/q.sum(1,keepdim=True).clamp_min(1e-6)
    flat=q.flatten(2).transpose(1,2)
    f=F.normalize(dense.flatten(2).transpose(1,2).float(),dim=-1)
    sim=f@f.transpose(1,2)
    n=h*w
    ids=sim.topk(min(topk,n),dim=-1).indices
    knn=torch.zeros_like(sim,dtype=torch.bool).scatter_(-1,ids,True)
    mutual=knn & knn.transpose(1,2)
    # Local links preserve thin structures; long-range links require mutual support.
    yy,xx=torch.meshgrid(torch.arange(h,device=q.device),torch.arange(w,device=q.device),indexing='ij')
    coord=torch.stack((yy.flatten(),xx.flatten()),1)
    nearby=(coord[:,None]-coord[None]).abs().amax(-1)<=1
    allowed=mutual | nearby[None]
    # Strongest class conflicts cannot provide propagation edges.
    confidence,winner=flat.max(-1)
    reliable=confidence>.65
    conflict=(winner[:,:,None]!=winner[:,None,:]) & reliable[:,:,None] & reliable[:,None,:]
    allowed=allowed & ~conflict
    allowed.diagonal(dim1=-2,dim2=-1).fill_(True)
    graph=((sim-1)/.10).exp()*allowed
    graph=graph/graph.sum(-1,keepdim=True).clamp_min(1e-8)
    anchors=flat.clone()
    propagated=flat
    for _ in range(steps):
        propagated=(1-strength)*flat+strength*(graph@propagated)
        propagated=torch.where(reliable[:,:,None],anchors,propagated)
    # DINO class prototypes use only confident CAM seeds from this image.
    seed_weight=flat*(flat>=flat.amax(1,keepdim=True)*.8)
    prototype=F.normalize(seed_weight.transpose(1,2)@f,dim=-1)
    proto_logits=(f@prototype.transpose(1,2))/.1
    present=torch.cat((torch.ones_like(labels[:,:1]),labels),1).bool()
    available=(seed_weight.sum(1)>1e-6)&present
    proto_logits=proto_logits.masked_fill(~available[:,None],-1e4)
    proto=proto_logits.softmax(-1).transpose(1,2).reshape(b,c+1,h,w)
    propagated=propagated.transpose(1,2).reshape(b,c+1,h,w)
    foreground=propagated[:,1:]
    foreground=foreground/foreground.flatten(2).amax(-1)[:,:,None,None].clamp_min(1e-6)
    return foreground,proto


def partial_label_loss(logits, pseudo, prototype, labels, min_conf=.7):
    """On credible disagreements, supervise the union of two plausible classes.

    No validation tags are used; image labels apply only to training targets.
    A rejected/absent prototype falls back to the original hard pseudo label.
    """
    proto=F.interpolate(prototype.detach(),pseudo.shape[-2:],mode='bilinear',align_corners=False)
    confidence,other=proto.max(1)
    present=torch.cat((torch.ones_like(labels[:,:1]),labels),1).bool()
    present_other=present.gather(1,other.flatten(1)).reshape_as(other)
    valid=pseudo!=255
    conflict=(confidence>=min_conf)&(other!=pseudo)&present_other&valid
    safe=pseudo.clamp(0,logits.shape[1]-1)
    lp=logits.log_softmax(1)
    primary=lp.gather(1,safe[:,None]).squeeze(1)
    alternate=lp.gather(1,other[:,None]).squeeze(1)
    set_logp=torch.logaddexp(primary,alternate)
    loss=-torch.where(conflict,set_logp,primary)
    # Keep the same foreground/background balance as the control CE loss.
    bg=valid&(pseudo==0); fg=valid&(pseudo!=0)
    value=.5*((loss*bg).sum()/bg.sum().clamp_min(1)+(loss*fg).sum()/fg.sum().clamp_min(1))
    return value,conflict.float().mean()
