import segmentation_models_pytorch as smp
import torch

x = torch.randn(1,3,224,224)
model = smp.Unet('resnet34')
pred = model(x)

print(pred, pred.shape)