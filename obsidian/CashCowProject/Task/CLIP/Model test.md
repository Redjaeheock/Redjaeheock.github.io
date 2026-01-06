https://github.com/openai/CLIP

## Usage

First, [install PyTorch 1.7.1](https://pytorch.org/get-started/locally/) (or later) and torchvision, as well as small additional dependencies, and then install this repo as a Python package. On a CUDA GPU machine, the following will do the trick

```
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
```

여느 모델 처럼 python 3.10 버전을 기준으로 설치하려다 버전 충돌이 발생했다
![[Pasted image 20251224082756.png]]
아무래도 CLIP 은 2025 기준 python 3.10 버전 지원은 안되는 것 같다



Replace `cudatoolkit=11.0` above with the appropriate CUDA version on your machine or `cpuonly` when installing on a machine without a GPU.

