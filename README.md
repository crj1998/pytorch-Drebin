## Hardware

CPU: 2  Intel(R) Xeon(R) CPU @ 2.20GHz
GPU: Nvidia Tesla T4 16GB

Note: GPU recommended. When training, CPU need about 30min per epoch, while GPU(T4) only need 2~3 min per epoch.


## Software
OS: Ubuntu 18.04.5
Python: 3.7.10
PyTorch: 1.8.1+cu101

## Dataset

The official website of Drebin Dataset `https://www.sec.cs.tu-bs.de/~danarp/drebin/download.html`

However, download dataset from official source need authorization, we will provide dataset directly.

[Download Drebin from Jbox](https://jbox.sjtu.edu.cn/v/link/view/e74e7f39538f430a8ef271fe65bf197a)



## Command
```
# unzip data to ./drebin
unzip drebin.zip > /dev/null

# install dependent packages
pip install -r requirements.txt

# choose torch version you need
# CPU version
# pip install torch==1.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
# GPU version
# pip install torch

# preprocess data which will create `feature.pkl` in ./drebin folder
python preprocess.py

# run the main with 6 epochs eps=4
python main.py --epochs 6 --eps 4
```

## Jupyter Lab
```
setup_seed(0)
net = Net(test_loader.num_features)
net.load_state_dict(torch.load(PATH))
net = net.to(device)
```
replace `PATH` with trained model, such as `AT.pth` or `ST.pth`. We also provide pre-trained models which can access by 
```
链接: https://pan.baidu.com/s/1pefyf8kKSmcEsfKy4Z1fGg 提取码: tc21 复制这段内容后打开百度网盘手机App，操作更方便哦
```
download `*.pth` files and put it in the current folder. 