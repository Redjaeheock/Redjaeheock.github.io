
#### set environment
```
(phi-4) redjh@red-laptop:~/phi_4_multi$ pip list
Package                  Version
------------------------ ------------
accelerate               1.3.0        (+add)
backoff                  2.2.1        (+add)
certifi                  2025.10.5
cffi                     2.0.0
charset-normalizer       3.4.4
einops                   0.8.1
filelock                 3.19.1
flash_attn               2.7.4.post1  (+add)
fsspec                   2025.9.0
hf-xet                   1.1.10
huggingface-hub          0.35.3
idna                     3.11
Jinja2                   3.1.6
MarkupSafe               2.1.5
mpmath                   1.3.0
networkx                 3.3
numpy                    2.1.2
nvidia-cublas-cu12       12.4.5.8
nvidia-cuda-cupti-cu12   12.4.127
nvidia-cuda-nvrtc-cu12   12.4.127
nvidia-cuda-runtime-cu12 12.4.127
nvidia-cudnn-cu12        9.1.0.70
nvidia-cufft-cu12        11.2.1.3
nvidia-curand-cu12       10.3.5.147
nvidia-cusolver-cu12     11.6.1.9
nvidia-cusparse-cu12     12.3.1.170
nvidia-cusparselt-cu12   0.6.2
nvidia-nccl-cu12         2.21.5
nvidia-nvjitlink-cu12    12.4.127
nvidia-nvtx-cu12         12.4.127
packaging                25.0
peft                     0.13.2        (+add)
pillow                   11.1.0        (+add)
pip                      25.2
psutil                   7.1.0
pycparser                2.23
PyYAML                   6.0.3
regex                    2025.9.18
requests                 2.32.5
safetensors              0.6.2
scipy                    1.15.2        (+add)
setuptools               80.9.0
soundfile                0.13.1        (+add)
sympy                    1.13.1
tokenizers               0.21.4
torch                    2.6.0+cu124   (+add)
torchvision              0.21.0+cu124  (+add)
tqdm                     4.67.1
transformers             4.48.2        (+add)
triton                   3.2.0
typing_extensions        4.15.0
urllib3                  2.5.0
wheel                    0.45.1
```

#### test_phi.py

```python
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch, os, time, json

# FlashAttention 비활성화 (그대로 유지)
os.environ["USE_FLASH_ATTENTION_2"] = "0"
os.environ["HF_USE_FLASH_ATTENTION_2"] = "0"
os.environ["FLASH_ATTENTION_SKIP_IMPORT"] = "1"

model_id = "microsoft/Phi-4-multimodal-instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("🔹 모델 로드 중...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
    attn_implementation="eager",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# ✅ 이미지 토큰 반드시 추가!
prompt = "이 이미지를 보고 기사 내용을 요약해 주세요. <|image_1|>"

image = Image.open("test_data/image/test_article1.png").convert("RGB")

inputs = processor(prompt, images=[image], return_tensors="pt").to(device)

print("🧠 추론 시작...")
start = time.time()
outputs = model.generate(**inputs, max_new_tokens=150)
end = time.time()

result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
print("✅ 결과:", result)
print(f"⏱️ 추론 시간: {end - start:.2f}초")

os.makedirs("outputs", exist_ok=True)
with open("outputs/result.json", "w", encoding="utf-8") as f:
    json.dump({"prompt": prompt, "result": result}, f, ensure_ascii=False, indent=2)
```


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
```
```
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



