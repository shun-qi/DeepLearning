#!/bin/bash
# 创建并配置深度学习环境

# 创建环境
conda create -n dl_models python=3.9 -y

# 激活环境
source activate dl_models

# 安装PyTorch (CPU版本)
# pip install torch torchvision torchaudio

# 如果需要GPU版本，取消下面注释并注释上面的CPU版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# 安装其他依赖
pip install numpy pandas scikit-learn matplotlib nltk pillow tqdm

# 下载NLTK数据
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

echo "环境设置完成！使用 'conda activate dl_models' 激活环境"