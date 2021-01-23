import torch
from glasses.models import AutoModel, AutoConfig, EfficientNetLite
import timm
from transfer_weights import clone_model
from benchmark import benchmark
from glasses.models.classification.vit import ViTTokens
from glasses.models.classification.deit import DeiT, DeiTTokens

# src = timm.create_model('deit_small_distilled_patch16_224', pretrained='True')
# dst = AutoModel.from_name('vit_base_patch16_224')



src = torch.hub.load('facebookresearch/deit:main', 'deit_base_distilled_patch16_224', pretrained=True)
dst = DeiT.deit_base_distilled_patch16_224().eval()

transform = AutoConfig.from_name('vit_base_patch16_224').transform

dst = clone_model(src, dst, torch.randn((1, 3, 224, 224)), dest_skip = [DeiTTokens]).eval()

dst.embedding.positions.data.copy_(src.pos_embed.data.squeeze(0))
dst.embedding.tokens.cls.data.copy_(src.cls_token.data)
dst.embedding.tokens.dist.data.copy_(src.dist_token.data)

f1 = benchmark(src.cuda().eval(), transform, batch_size=128)
print(f1)