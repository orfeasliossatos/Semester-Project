import torch
if torch.cuda.is_available():
    print("Cuda is available")
else:
    print("Cuda is not available")

print("Torch version : ", torch.__version__)

print(torch.zeros(1).cuda())


