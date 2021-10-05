from glasses.models.classification.vovnet import *
from torchinfo import summary
import timm
from transfer_weights import clone_model, deit_clone
from benchmark import benchmark
from glasses.models import AutoTransform
import timm
from pprint import pprint
from pytorchcv.model_provider import get_model as ptcv_get_model
from einops.layers.torch import Rearrange, Reduce
from glasses.nn.blocks.residuals import ResidualAdd


# model_names = timm.list_models(pretrained=True)
# pprint(model_names)
m=VoVNetBlock(128, 128, 64)
# m = VoVNetBlock(128, 128, 64)
out=m(torch.randn((3, 128, 58, 58)))
print(out.shape)

# m = VoVNet.vovnet39(block=VoVNetV2Block)(torch.randn((1, 3, 224, 224)))
# m = VoVNetLayer(128, 128, stage_features=64, depth=2)
# out = m(torch.randn((3, 128, 58, 58)))
# model_names=timm.list_models("*vovnet**")
# pprint(model_names)
with torch.no_grad():
# # pcv_m = ptcv_get_model("vovnet27s", pretrained=True).cuda()
# # print(pcv_m)
    timm_m = timm.create_model("ese_vovnet39b", pretrained=True).cuda()
    m = VoVNet.vovnet39(block=VoVNetV2Block).cuda()
    m.summary()
    summary(timm_m, input_size=(1, 3, 224, 224), depth=-1)
    m = clone_model(timm_m, m, torch.rand((1, 3, 224, 224)).cuda(), verbose=False)
    res = benchmark(
        m,
        AutoTransform.from_name("resnet18"),
        device=torch.device("cuda"),
        fast=True,
    )
    print(res)

# summary(m, input_size=(1, 3, 224, 224))
