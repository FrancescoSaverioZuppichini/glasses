from glasses.models.classification.vovnet import *
m = VoVNetBlock(128, 128, 64)
out = m(torch.randn((3, 128, 58, 58)))
print(out.shape)

m = VoVNetLayer(128, 128, stage_features=64)
out = m(torch.randn((3, 128, 58, 58)))
print(out.shape)

print(VoVNetStem(3))