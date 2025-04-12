import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from PIL import Image

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("所有依赖已成功安装！")