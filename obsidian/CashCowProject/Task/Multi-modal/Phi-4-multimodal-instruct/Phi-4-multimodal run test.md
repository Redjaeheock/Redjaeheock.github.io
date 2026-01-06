
# Requirement

| 패키지                         | 역할 / 설명                                                  |
| --------------------------- | -------------------------------------------------------- |
| **flash-attn==2.7.4.post1** | FlashAttention 커널 가속. A100/H100용. <br>(MX450 등에서는 사용 불가) |
| **torch==2.6.0**            | PyTorch 핵심 연산 라이브러리                                      |
| **transformers==4.48.2**    | Phi-4 계열이 통합된 최신 버전 (4.48.0 이상 필수)                       |
| **accelerate==1.3.0**       | CPU/GPU 자동 분배 및 mixed precision 지원                       |
| **soundfile==0.13.1**       | 멀티모달 오디오 입력용                                             |
| **pillow==11.1.0**          | 이미지 입출력용                                                 |
| **scipy==1.15.2**           | 신호처리 및 오디오 변환 시 사용                                       |
| **torchvision==0.21.0**     | 비전 모듈용                                                   |
| **backoff==2.2.1**          | 내부 API 재시도용 유틸리티                                         |
| **peft==0.13.2**            | LoRA 가중치 통합용                                             |

# First Test - basic
(flash-attn 미설치)
#### phi_４.py
```python
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
from PIL import Image
import torch
import json
import time

# ====== 설정 ======
model_id = "microsoft/Phi-4-multimodal-instruct"

device = "cuda" if torch.cuda.is_available() else "cpu" # MX450 OOM 발생

# ====== 모델 로드 ======
print("🔹 모델 로드 중...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
    attn_implementation="eager",  # FlashAttention 비활성화
    trust_remote_code=True,       # 보안 확인 프롬프트 y 처리
)
processor = AutoProcessor.from_pretrained(model_id)

# ====== 입력 ======
prompt = "이 이미지를 보고 기사 내용을 요약해줘. <|image_1|>"
image = Image.open("test_data/image/test_article1.png").resize((512,512))

inputs = processor(prompt, images=image, return_tensors="pt").to(device)

# ====== 추론 ======
print("🧠 추론 시작...")
start = time.time()
outputs = model.generate(**inputs, max_new_tokens=200)
end = time.time()

# ====== 출력 ======
result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
print("✅ 결과:", result)
print(f"⏱️ 추론 시간: {end - start:.2f}초")

# ====== 저장 ======
_os.makedirs("outputs", exist_ok=True)
with open("outputs/result.json", "w", encoding="utf-8") as f:
    json.dump({"prompt": prompt, "result": result}, f, ensure_ascii=False, indent=2)
```

### 결과
![[Pasted image 20251006130539.png]]
(Error 내용 확대)
![[Pasted image 20251006130732.png]]
FlashAttention 라이브러리를 설치하지 않았지만, 모델 자체에서 해당 라이브러리를 호출해서 상요하기 때문에 
``` python
attn_implementation="eager"
```
설정을 해도 넘어가 지지가 않는다.

### Solution 1

flash-atten 를 설치만 하고, 사용 설정은 그대로 꺼두기

![[Pasted image 20251006131127.png]]
문제가 생겼길래 알아 보니 **CUDA Toolkit**의 부재로 발생했다고 한다.
FlashAttention 은 PyPI 패키지 이지만 
**실제로는 'CUDA C++' 커널을 로컬에서 컴파일해야 하는 소스 배포형 패키지'** 라서 
CUDA Toolkit 을 요구한다
(이 글 작성시점에서 Toolkit 버전은 11.7 을 요구하고 있음)


```
이 시점에서 windows 에 그래픽 드라이버 설치와 tooklit 설치 및 wsl 에서 드라이버 설치와 tooklit 설치에 관해 알아보고 정리해보았다

결론: 
windows 사용자는 window 에 드라이버를 설치하고,

GPU 작업이 windows OS 에서 이뤄진다면 windows 에다가 toolkit 을 설치하고,
GPU 작업이 WSL 에서 이뤄진다면 WSL 에다가 toolkit 을 설치한다

어떤 쪽에 toolkit 을 사용하더라도, 그 상위는 windows 에 설치한 드라이버를 통해서 작업이 이뤄진다
```


### Solution 1-1. GPU 드라이버 및 toolkit 재설정 후 flash_attn 설치

```
(phi-4) redjh@red-laptop:~$ pip install flash_attn==2.7.4.post1
Collecting flash_attn==2.7.4.post1
  Using cached flash_attn-2.7.4.post1.tar.gz (6.0 MB)
  Preparing metadata (setup.py) ... done
Requirement already satisfied: torch in ./miniconda3/envs/phi-4/lib/python3.10/site-packages (from flash_attn==2.7.4.post1) (2.6.0+cu124)
Collecting einops (from flash_attn==2.7.4.post1)
  Downloading einops-0.8.1-py3-none-any.whl.metadata (13 kB)
Requirement already satisfied: filelock in ./miniconda3/envs/phi-4/lib/python3.10/site-packages (from torch->flash_attn==2.7.4.post1) (3.19.1)
Requirement already satisfied: typing-extensions>=4.10.0 in ./miniconda3/envs/phi-4/lib/python3.10/site-packages (from torch->flash_attn==2.7.4.post1) (4.15.0)
Requirement already satisfied: networkx in ./miniconda3/envs/phi-4/lib/python3.10/site-packages (from torch->flash_attn==2.7.4.post1) (3.3)
Requirement already satisfied: jinja2 in ./miniconda3/envs/phi-4/lib/python3.10/site-packages (from torch->flash_attn==2.7.4.post1) (3.1.6)
Requirement already satisfied: fsspec in ./miniconda3/envs/phi-4/lib/python3.10/site-packages (from torch->flash_attn==2.7.4.post1) (2025.9.0)
Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.4.127 in ./miniconda3/envs/phi-4/lib/python3.10/site-packages (from torch->flash_attn==2.7.4.post1) (12.4.127)
Requirement already satisfied: nvidia-cuda-runtime-cu12==12.4.127 in ./miniconda3/envs/phi-4/lib/python3.10/site-packages (from torch->flash_attn==2.7.4.post1) (12.4.127)
Requirement already satisfied: nvidia-cuda-cupti-cu12==12.4.127 in ./miniconda3/envs/phi-4/lib/python3.10/site-packages (from torch->flash_attn==2.7.4.post1) (12.4.127)
Requirement already satisfied: nvidia-cudnn-cu12==9.1.0.70 in ./miniconda3/envs/phi-4/lib/python3.10/site-packages (from torch->flash_attn==2.7.4.post1) (9.1.0.70)
Requirement already satisfied: nvidia-cublas-cu12==12.4.5.8 in ./miniconda3/envs/phi-4/lib/python3.10/site-packages (from torch->flash_attn==2.7.4.post1) (12.4.5.8)
Requirement already satisfied: nvidia-cufft-cu12==11.2.1.3 in ./miniconda3/envs/phi-4/lib/python3.10/site-packages (from torch->flash_attn==2.7.4.post1) (11.2.1.3)
Requirement already satisfied: nvidia-curand-cu12==10.3.5.147 in ./miniconda3/envs/phi-4/lib/python3.10/site-packages (from torch->flash_attn==2.7.4.post1) (10.3.5.147)
Requirement already satisfied: nvidia-cusolver-cu12==11.6.1.9 in ./miniconda3/envs/phi-4/lib/python3.10/site-packages (from torch->flash_attn==2.7.4.post1) (11.6.1.9)
Requirement already satisfied: nvidia-cusparse-cu12==12.3.1.170 in ./miniconda3/envs/phi-4/lib/python3.10/site-packages (from torch->flash_attn==2.7.4.post1) (12.3.1.170)
Requirement already satisfied: nvidia-cusparselt-cu12==0.6.2 in ./miniconda3/envs/phi-4/lib/python3.10/site-packages (from torch->flash_attn==2.7.4.post1) (0.6.2)
Requirement already satisfied: nvidia-nccl-cu12==2.21.5 in ./miniconda3/envs/phi-4/lib/python3.10/site-packages (from torch->flash_attn==2.7.4.post1) (2.21.5)
Requirement already satisfied: nvidia-nvtx-cu12==12.4.127 in ./miniconda3/envs/phi-4/lib/python3.10/site-packages (from torch->flash_attn==2.7.4.post1) (12.4.127)
Requirement already satisfied: nvidia-nvjitlink-cu12==12.4.127 in ./miniconda3/envs/phi-4/lib/python3.10/site-packages (from torch->flash_attn==2.7.4.post1) (12.4.127)
Requirement already satisfied: triton==3.2.0 in ./miniconda3/envs/phi-4/lib/python3.10/site-packages (from torch->flash_attn==2.7.4.post1) (3.2.0)
Requirement already satisfied: sympy==1.13.1 in ./miniconda3/envs/phi-4/lib/python3.10/site-packages (from torch->flash_attn==2.7.4.post1) (1.13.1)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in ./miniconda3/envs/phi-4/lib/python3.10/site-packages (from sympy==1.13.1->torch->flash_attn==2.7.4.post1) (1.3.0)
Requirement already satisfied: MarkupSafe>=2.0 in ./miniconda3/envs/phi-4/lib/python3.10/site-packages (from jinja2->torch->flash_attn==2.7.4.post1) (2.1.5)
Downloading einops-0.8.1-py3-none-any.whl (64 kB)
Building wheels for collected packages: flash_attn
  DEPRECATION: Building 'flash_attn' using the legacy setup.py bdist_wheel mechanism, which will be removed in a future version. pip 25.3 will enforce this behaviour change. A possible replacement is to use the standardized build interface by setting the `--use-pep517` option, (possibly combined with `--no-build-isolation`), or adding a `pyproject.toml` file to the source tree of 'flash_attn'. Discussion can be found at https://github.com/pypa/pip/issues/6334
  Building wheel for flash_attn (setup.py) ... done
  Created wheel for flash_attn: filename=flash_attn-2.7.4.post1-cp310-cp310-linux_x86_64.whl size=187815087 sha256=ffe17686fa1a0f288de9eae7c32af209d32a27b037ef28614f042b377af5b15a
  Stored in directory: /home/redjh/.cache/pip/wheels/59/ce/d5/08ea07bfc16ba218dc65a3a7ef9b6a270530bcbd2cea2ee1ca
Successfully built flash_attn
Installing collected packages: einops, flash_attn
Successfully installed einops-0.8.1 flash_attn-2.7.4.post1
```

flash-attn 설치 후 재실행

```
🔹 모델 로드 중...
config.json: 4.63kB [00:00, 11.4MB/s]
configuration_phi4mm.py: 11.0kB [00:00, 37.0MB/s]
A new version of the following files was downloaded from https://huggingface.co/microsoft/Phi-4-multimodal-instruct:
- configuration_phi4mm.py
. Make sure to double-check they do not contain any added malicious code. To avoid downloading new versions of the code file, you can pin a revision.
modeling_phi4mm.py: 116kB [00:00, 50.2MB/s]
vision_siglip_navit.py: 78.2kB [00:00, 103MB/s]
A new version of the following files was downloaded from https://huggingface.co/microsoft/Phi-4-multimodal-instruct:
- vision_siglip_navit.py
. Make sure to double-check they do not contain any added malicious code. To avoid downloading new versions of the code file, you can pin a revision.
processing_phi4mm.py: 32.8kB [00:00, 59.3MB/s]
A new version of the following files was downloaded from https://huggingface.co/microsoft/Phi-4-multimodal-instruct:
- processing_phi4mm.py
. Make sure to double-check they do not contain any added malicious code. To avoid downloading new versions of the code file, you can pin a revision.
speech_conformer_encoder.py: 111kB [00:00, 54.6MB/s]
A new version of the following files was downloaded from https://huggingface.co/microsoft/Phi-4-multimodal-instruct:
- speech_conformer_encoder.py
. Make sure to double-check they do not contain any added malicious code. To avoid downloading new versions of the code file, you can pin a revision.
A new version of the following files was downloaded from https://huggingface.co/microsoft/Phi-4-multimodal-instruct:
- modeling_phi4mm.py
- vision_siglip_navit.py
- processing_phi4mm.py
- speech_conformer_encoder.py
. Make sure to double-check they do not contain any added malicious code. To avoid downloading new versions of the code file, you can pin a revision.
/home/redjh/miniconda3/envs/phi-4/lib/python3.10/site-packages/transformers/models/auto/image_processing_auto.py:590: FutureWarning: The image_processor_class argument is deprecated and will be removed in v4.42. Please use `slow_image_processor_class`, or `fast_image_processor_class` instead
  warnings.warn(
model.safetensors.index.json: 240kB [00:00, 142MB/s]
model-00001-of-00003.safetensors: 100%|████████████████████████████████████████████████████████████████████████████████| 5.00G/5.00G [04:37<00:00, 18.0MB/s]
model-00002-of-00003.safetensors: 100%|████████████████████████████████████████████████████████████████████████████████| 4.95G/4.95G [04:29<00:00, 18.4MB/s]
model-00003-of-00003.safetensors: 100%|████████████████████████████████████████████████████████████████████████████████| 1.20G/1.20G [01:07<00:00, 17.8MB/s]
Downloading shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [10:15<00:00, 205.33s/it]
/home/redjh/.cache/huggingface/modules/transformers_modules/microsoft/Phi-4-multimodal-instruct/33e62acdd07cd7d6635badd529aa0a3467bb9c6a/speech_conformer_encoder.py:2774: FutureWarning: Please specify CheckpointImpl.NO_REENTRANT as CheckpointImpl.REENTRANT will soon be removed as the default and eventually deprecated.
  lambda i: encoder_checkpoint_wrapper(
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:18<00:00,  6.08s/it]
generation_config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 190/190 [00:00<00:00, 1.42MB/s]
Some parameters are on the meta device because they were offloaded to the disk and cpu.
processor_config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 121/121 [00:00<00:00, 1.23MB/s]
preprocessor_config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 482/482 [00:00<00:00, 4.35MB/s]
Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.48, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
tokenizer_config.json: 3.25kB [00:00, 2.53MB/s]
vocab.json: 3.91MB [00:00, 31.6MB/s]
merges.txt: 2.42MB [00:00, 33.0MB/s]
tokenizer.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 15.5M/15.5M [00:02<00:00, 7.58MB/s]
added_tokens.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 249/249 [00:00<00:00, 1.12MB/s]
special_tokens_map.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 473/473 [00:00<00:00, 3.63MB/s]
```
```
🧠 추론 시작...
/home/redjh/miniconda3/envs/phi-4/lib/python3.10/site-packages/transformers/generation/utils.py:2137: UserWarning: You are calling .generate() with the `input_ids` being on a device type different than your model's device. `input_ids` is on cuda, whereas the model is on cpu. You may experience unexpected behaviors or slower generation. Please make sure that you have put `input_ids` to the correct device by calling for example input_ids = input_ids.to('cpu') before running `.generate()`.
  warnings.warn(
Traceback (most recent call last):
  File "/home/redjh/phi_4_multi/phi_4.py", line 32, in <module>
    outputs = model.generate(**inputs, max_new_tokens=150)
  File "/home/redjh/miniconda3/envs/phi-4/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
  File "/home/redjh/miniconda3/envs/phi-4/lib/python3.10/site-packages/transformers/generation/utils.py", line 2255, in generate
    result = self._sample(
  File "/home/redjh/miniconda3/envs/phi-4/lib/python3.10/site-packages/transformers/generation/utils.py", line 3254, in _sample
    outputs = self(**model_inputs, return_dict=True)
  File "/home/redjh/miniconda3/envs/phi-4/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/redjh/miniconda3/envs/phi-4/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/redjh/miniconda3/envs/phi-4/lib/python3.10/site-packages/accelerate/hooks.py", line 170, in new_forward
    output = module._old_forward(*args, **kwargs)
  File "/home/redjh/.cache/huggingface/modules/transformers_modules/microsoft/Phi-4-multimodal-instruct/33e62acdd07cd7d6635badd529aa0a3467bb9c6a/modeling_phi4mm.py", line 2116, in forward
    outputs = self.model(
  File "/home/redjh/miniconda3/envs/phi-4/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/redjh/miniconda3/envs/phi-4/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/redjh/.cache/huggingface/modules/transformers_modules/microsoft/Phi-4-multimodal-instruct/33e62acdd07cd7d6635badd529aa0a3467bb9c6a/modeling_phi4mm.py", line 1707, in forward
    inputs_embeds = self.embed_tokens_extend(
  File "/home/redjh/miniconda3/envs/phi-4/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/redjh/miniconda3/envs/phi-4/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/redjh/miniconda3/envs/phi-4/lib/python3.10/site-packages/accelerate/hooks.py", line 170, in new_forward
    output = module._old_forward(*args, **kwargs)
  File "/home/redjh/.cache/huggingface/modules/transformers_modules/microsoft/Phi-4-multimodal-instruct/33e62acdd07cd7d6635badd529aa0a3467bb9c6a/modeling_phi4mm.py", line 769, in forward
    image_hidden_states = self.image_embed(
  File "/home/redjh/miniconda3/envs/phi-4/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/redjh/miniconda3/envs/phi-4/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/redjh/miniconda3/envs/phi-4/lib/python3.10/site-packages/accelerate/hooks.py", line 170, in new_forward
    output = module._old_forward(*args, **kwargs)
  File "/home/redjh/.cache/huggingface/modules/transformers_modules/microsoft/Phi-4-multimodal-instruct/33e62acdd07cd7d6635badd529aa0a3467bb9c6a/modeling_phi4mm.py", line 328, in forward
    img_features = self.get_img_features(img_embeds.flatten(0, 1), attention_mask=image_attention_mask.type(torch.BoolTensor).flatten(0,1).to(target_device))
  File "/home/redjh/.cache/huggingface/modules/transformers_modules/microsoft/Phi-4-multimodal-instruct/33e62acdd07cd7d6635badd529aa0a3467bb9c6a/modeling_phi4mm.py", line 194, in get_img_features
    img_processor_output = self.img_processor(img_embeds, output_hidden_states=True, patch_attention_mask=attention_mask)
  File "/home/redjh/miniconda3/envs/phi-4/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/redjh/miniconda3/envs/phi-4/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/redjh/miniconda3/envs/phi-4/lib/python3.10/site-packages/accelerate/hooks.py", line 170, in new_forward
    output = module._old_forward(*args, **kwargs)
  File "/home/redjh/.cache/huggingface/modules/transformers_modules/microsoft/Phi-4-multimodal-instruct/33e62acdd07cd7d6635badd529aa0a3467bb9c6a/vision_siglip_navit.py", line 1385, in forward
    encoder_outputs = self.encoder(
  File "/home/redjh/miniconda3/envs/phi-4/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/redjh/miniconda3/envs/phi-4/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/redjh/miniconda3/envs/phi-4/lib/python3.10/site-packages/accelerate/hooks.py", line 170, in new_forward
    output = module._old_forward(*args, **kwargs)
  File "/home/redjh/.cache/huggingface/modules/transformers_modules/microsoft/Phi-4-multimodal-instruct/33e62acdd07cd7d6635badd529aa0a3467bb9c6a/vision_siglip_navit.py", line 1179, in forward
    layer_outputs = encoder_layer(
  File "/home/redjh/miniconda3/envs/phi-4/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/redjh/miniconda3/envs/phi-4/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/redjh/miniconda3/envs/phi-4/lib/python3.10/site-packages/accelerate/hooks.py", line 170, in new_forward
    output = module._old_forward(*args, **kwargs)
  File "/home/redjh/.cache/huggingface/modules/transformers_modules/microsoft/Phi-4-multimodal-instruct/33e62acdd07cd7d6635badd529aa0a3467bb9c6a/vision_siglip_navit.py", line 953, in forward
    hidden_states, attn_weights = self.self_attn(
  File "/home/redjh/miniconda3/envs/phi-4/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/redjh/miniconda3/envs/phi-4/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/redjh/miniconda3/envs/phi-4/lib/python3.10/site-packages/accelerate/hooks.py", line 170, in new_forward
    output = module._old_forward(*args, **kwargs)
  File "/home/redjh/.cache/huggingface/modules/transformers_modules/microsoft/Phi-4-multimodal-instruct/33e62acdd07cd7d6635badd529aa0a3467bb9c6a/vision_siglip_navit.py", line 797, in forward
    attn_output = self._flash_attention_forward(
  File "/home/redjh/.cache/huggingface/modules/transformers_modules/microsoft/Phi-4-multimodal-instruct/33e62acdd07cd7d6635badd529aa0a3467bb9c6a/vision_siglip_navit.py", line 844, in _flash_attention_forward
    attn_output_unpad = flash_attn_varlen_func(
  File "/home/redjh/miniconda3/envs/phi-4/lib/python3.10/site-packages/flash_attn/flash_attn_interface.py", line 1448, in flash_attn_varlen_func
    return FlashAttnVarlenFunc.apply(
  File "/home/redjh/miniconda3/envs/phi-4/lib/python3.10/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
  File "/home/redjh/miniconda3/envs/phi-4/lib/python3.10/site-packages/flash_attn/flash_attn_interface.py", line 930, in forward
    out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_varlen_forward(
  File "/home/redjh/miniconda3/envs/phi-4/lib/python3.10/site-packages/torch/_ops.py", line 1123, in __call__
    return self._op(*args, **(kwargs or {}))
  File "/home/redjh/miniconda3/envs/phi-4/lib/python3.10/site-packages/torch/_library/autograd.py", line 113, in autograd_impl
    result = forward_no_grad(*args, Metadata(keyset, keyword_only_args))
  File "/home/redjh/miniconda3/envs/phi-4/lib/python3.10/site-packages/torch/_library/autograd.py", line 40, in forward_no_grad
    result = op.redispatch(keyset & _C._after_autograd_keyset, *args, **kwargs)
  File "/home/redjh/miniconda3/envs/phi-4/lib/python3.10/site-packages/torch/_ops.py", line 728, in redispatch
    return self._handle.redispatch_boxed(keyset, *args, **kwargs)
NotImplementedError: Could not run 'flash_attn::_flash_attn_varlen_forward' with arguments from the 'CPU' backend. This could be because the operator doesn't exist for this backend, or was omitted during the selective/custom build process (if using custom build). If you are a Facebook employee using PyTorch on mobile, please visit https://fburl.com/ptmfixes for possible resolutions. 'flash_attn::_flash_attn_varlen_forward' is only available for these backends: [CUDA, Meta, BackendSelect, Python, FuncTorchDynamicLayerBackMode, Functionalize, Named, Conjugate, Negative, ZeroTensor, ADInplaceOrView, AutogradOther, AutogradCPU, AutogradCUDA, AutogradHIP, AutogradXLA, AutogradMPS, AutogradIPU, AutogradXPU, AutogradHPU, AutogradVE, AutogradLazy, AutogradMTIA, AutogradPrivateUse1, AutogradPrivateUse2, AutogradPrivateUse3, AutogradMeta, AutogradNestedTensor, Tracer, AutocastCPU, AutocastXPU, AutocastMPS, AutocastCUDA, FuncTorchBatched, BatchedNestedTensor, FuncTorchVmapMode, Batched, VmapMode, FuncTorchGradWrapper, PythonTLSSnapshot, FuncTorchDynamicLayerFrontMode, PreDispatch, PythonDispatcher].

CUDA: registered at /dev/null:173 [kernel]
Meta: registered at /dev/null:198 [kernel]
BackendSelect: fallthrough registered at /pytorch/aten/src/ATen/core/BackendSelectFallbackKernel.cpp:3 [backend fallback]
Python: registered at /pytorch/aten/src/ATen/core/PythonFallbackKernel.cpp:194 [backend fallback]
FuncTorchDynamicLayerBackMode: registered at /pytorch/aten/src/ATen/functorch/DynamicLayer.cpp:503 [backend fallback]
Functionalize: registered at /pytorch/aten/src/ATen/FunctionalizeFallbackKernel.cpp:349 [backend fallback]
Named: registered at /pytorch/aten/src/ATen/core/NamedRegistrations.cpp:7 [backend fallback]
Conjugate: registered at /pytorch/aten/src/ATen/ConjugateFallback.cpp:17 [backend fallback]
Negative: registered at /pytorch/aten/src/ATen/native/NegateFallback.cpp:18 [backend fallback]
ZeroTensor: registered at /pytorch/aten/src/ATen/ZeroTensorFallback.cpp:86 [backend fallback]
ADInplaceOrView: fallthrough registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:100 [backend fallback]
AutogradOther: registered at /dev/null:173 [autograd kernel]
AutogradCPU: registered at /dev/null:173 [autograd kernel]
AutogradCUDA: registered at /dev/null:173 [autograd kernel]
AutogradHIP: registered at /dev/null:173 [autograd kernel]
AutogradXLA: registered at /dev/null:173 [autograd kernel]
AutogradMPS: registered at /dev/null:173 [autograd kernel]
AutogradIPU: registered at /dev/null:173 [autograd kernel]
AutogradXPU: registered at /dev/null:173 [autograd kernel]
AutogradHPU: registered at /dev/null:173 [autograd kernel]
AutogradVE: registered at /dev/null:173 [autograd kernel]
AutogradLazy: registered at /dev/null:173 [autograd kernel]
AutogradMTIA: registered at /dev/null:173 [autograd kernel]
AutogradPrivateUse1: registered at /dev/null:173 [autograd kernel]
AutogradPrivateUse2: registered at /dev/null:173 [autograd kernel]
AutogradPrivateUse3: registered at /dev/null:173 [autograd kernel]
AutogradMeta: registered at /dev/null:173 [autograd kernel]
AutogradNestedTensor: registered at /dev/null:173 [autograd kernel]
Tracer: registered at /pytorch/torch/csrc/autograd/TraceTypeManual.cpp:294 [backend fallback]
AutocastCPU: fallthrough registered at /pytorch/aten/src/ATen/autocast_mode.cpp:322 [backend fallback]
AutocastXPU: fallthrough registered at /pytorch/aten/src/ATen/autocast_mode.cpp:465 [backend fallback]
AutocastMPS: fallthrough registered at /pytorch/aten/src/ATen/autocast_mode.cpp:209 [backend fallback]
AutocastCUDA: fallthrough registered at /pytorch/aten/src/ATen/autocast_mode.cpp:165 [backend fallback]
FuncTorchBatched: registered at /pytorch/aten/src/ATen/functorch/LegacyBatchingRegistrations.cpp:731 [backend fallback]
BatchedNestedTensor: registered at /pytorch/aten/src/ATen/functorch/LegacyBatchingRegistrations.cpp:758 [backend fallback]
FuncTorchVmapMode: fallthrough registered at /pytorch/aten/src/ATen/functorch/VmapModeRegistrations.cpp:27 [backend fallback]
Batched: registered at /pytorch/aten/src/ATen/LegacyBatchingRegistrations.cpp:1075 [backend fallback]
VmapMode: fallthrough registered at /pytorch/aten/src/ATen/VmapModeRegistrations.cpp:33 [backend fallback]
FuncTorchGradWrapper: registered at /pytorch/aten/src/ATen/functorch/TensorWrapper.cpp:207 [backend fallback]
PythonTLSSnapshot: registered at /pytorch/aten/src/ATen/core/PythonFallbackKernel.cpp:202 [backend fallback]
FuncTorchDynamicLayerFrontMode: registered at /pytorch/aten/src/ATen/functorch/DynamicLayer.cpp:499 [backend fallback]
PreDispatch: registered at /pytorch/aten/src/ATen/core/PythonFallbackKernel.cpp:206 [backend fallback]
PythonDispatcher: registered at /pytorch/aten/src/ATen/core/PythonFallbackKernel.cpp:198 [backend fallback]
```

추론에서 오류가 발생

코드 내부에서 attn_flash 호출을 제약 시켰음에도 불구하고 attn_flash 호출로 인한 에러가 발생

\- `attn_implementation="eager"`의 의미

`transformers`의 모든 모델은 attention 계산 방식을 다음 3가지 중 하나로 설정할 수 있다:

| 옵션                    | 의미                              | FlashAttention 사용 여부 |
| --------------------- | ------------------------------- | -------------------- |
| `"flash_attention_2"` | FlashAttention v2 사용            | 사용                   |
| `"flash_attention"`   | FlashAttention v1 사용            | 사용                   |
| `"eager"`             | PyTorch의 기본 attention kernel 사용 | 사용 안 함               |

flash_attention 은 **'SM80 아키텍처 코어(Ampere)'** 부터 지원이 가능한 kernel 이기 때문에, 
현재 local 의 MX450(SM75, Turing) 에서는 사용할 수 없는 구조라 기본 kernel 을 사용하기 위해 'eager' 모드로 설정해야 한다.


ChatGPT 를 통해 해당 오류를 확인해보았다.
```
File ".../vision_siglip_navit.py", line 844, in _flash_attention_forward
    attn_output_unpad = flash_attn_varlen_func(
...
NotImplementedError: Could not run 'flash_attn::_flash_attn_varlen_forward' with arguments from the 'CPU' backend.
```
여기서 호출된 건 **`vision_siglip_navit.py` 내부의 `_flash_attention_forward()`** 라고 한다

- **Phi-4의 텍스트 모델 부분이 아니라, 비전 인코더(`SigLIP-NaViT`) 쪽**에서 FlashAttention을 시도

```
Phi-4-multimodal-instruct

Phi4MultiModalModel
├── text_model (Phi4ForCausalLM)
│     └── attn_implementation="eager"    <- 적용됨
└── image_model (SigLIP-NaViT backbone)
      └── vision encoder layers (use flash_attn by default not overridden) <- 문제
```

요약하자면

- 텍스트용 Transformer 에는 "eager" 가 적용되어 attn_flash 비활성화 됨
	
	- 일반 PyTorch의 `scaled_dot_product_attention`으로 동작.
	
- **비전 인코더**(이미지 인베딩) **SigLIP-NaViT** 쪽은 별도의 설정을 가지고 있어 따로 설정이 필요
	
	- `config.vision_config` 통해 설정
	- `attn_implementation` 필드가 아예 없거나 
	  `"flash_attention_2"`로 기본값 설정이 된 것으로 유추할 수 있다고 함
	- 자료형에 따라 각각의 flash_attn 설정이 존재하는 것으로 유추
		- 텍스트, 이미지, 오디오, 비디오
	
- 그래서 이미지 처리 쪽에는 "eager" 처리가 적용이 되지 않음
	
	- 그로인해 `SigLIP-NaViT`는 자체적으로 `flash_attn` 모듈을 import하고,
	  내부 함수 `_flash_attention_forward()`에서 FlashAttention CUDA 커널을 호출함


현재 모델이 CPU에 있고,  FlashAttention은 GPU 전용 커널이므로, CPU에서 이 함수가 호출되니
 
```
NotImplementedError: Could not run 'flash_attn::_flash_attn_varlen_forward' with arguments from the 'CPU' backend.
```
이와 같은 오류가 동반한 상황이라고 보고 있다.



### Solution 1-2. 환경 변수와 config 제어를 통해 해결

ChatGPT 와 토론을 통해 도출한 솔루션을 적용시켜 보고자 한다
### 해결 방법 및 원리

1. **FlashAttention 모듈 import 자체를 차단**  
    - `sys.modules["flash_attn"] = None`, 환경변수 비활성화 
	    - (모델 로드시 `flash_attn` import 불가)
    
2. **Vision 인코더 config에도 `"eager"` 직접 주입**  
    - `cfg.vision_config.attn_implementation = "eager"`
	    - (FlashAttention 경로가 `_flash_attention_forward()`에서 `eager`로 대체됨)


실제로 try 해보니 **1번 솔루션 단일**로는 해결이 되지 않았지만.
**2번 솔루션을 단일**로 사용하니 flash_attn 호출 인한 에러가 없어진 것을 확인될 수 있었다.



---
	