import torch
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
import timm.models.mlp_mixer as mm
import timm.models.vision_transformer as vit

inputs = (torch.randn((1,3,224,224)),)


#model = mm.mixer_ti16_224()
model = vit.vit_tiny_patch16_224()

flops = FlopCountAnalysis(model, inputs)
print(flop_count_table(flops))
