import torch

dja = torch.load("logs/W8A8_calib1024_batch64_iterW20000/resnet18/default.pth")


for k, v in dja.items():
    print(k, v.shape)
    if "act_quantizer" in k:
        print(v)
