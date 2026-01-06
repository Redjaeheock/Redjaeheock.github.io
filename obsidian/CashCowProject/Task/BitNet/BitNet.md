대규모 언어 모델을 위해 설계된 확장 가능하고 안정적인 1비트 트랜스포머 아키텍처인 BitNet

언어 모델링 실험 결과는 BitNet이 최첨단 8비트 양자화 방식 및 FP16 트랜스포머 베이스라인과 비교하여 메모리 사용량과 에너지 소비를 크게 줄이면서도 경쟁력 있는 성능을 달성함을 보여줌.
또한, BitNet은 완전 정밀도 트랜스포머와 유사한 스케일링 법칙을 나타내므로, 효율성과 성능 이점을 유지하면서 더 큰 규모의 언어 모델로 효과적으로 스케일링할 수 있는 잠재력을 시사함

---
[ gpt5 요약 ]

### 1. **BitNet**: 1-비트 트랜스포머 기반 대규모 언어 모델 (LLM)

- **원 논문**: _BitNet: Scaling 1‑bit Transformers for Large Language Models_ (2023년 10월)  
    → 트랜스포머 모델의 `nn.Linear` 레이어를 `BitLinear` 레이어로 대체하여, 가중치를 1‑비트로 학습하도록 설계된 구조입니다. 학습 과정에서의 추정은 고정소수점 연산을 유지하면서도 가중치를 1‑비트로 제한하여 추론 시 메모리와 에너지, 연산 시간을 크게 절감할 수 있습니다. [위키백과+15arXiv+15GitHub+15](https://arxiv.org/abs/2310.11453?utm_source=chatgpt.com)
    
- **BitNet b1.58 (2B4T)**: 마이크로소프트 연구진이 발표한 최신 버전  
    → `1.58비트`는 세 가지 값 {‑1, 0, +1}을 이용한 터너리 표현의 정보량(log₂3 ≈ 1.58비트)을 의미합니다.  
    → 이 모델은 순수하게 1.58비트로 훈련되었으며, 약 2억 개의 파라미터와 4조 개 토큰으로 학습됨. 전체 가중치가 공개되어 있으며, CPU/GPU에서 효율적으로 추론할 수 있는 코드를 포함합니다. 성능은 유사 크기의 full-precision LLM과 비교해도 경쟁력이 있습니다. [위키백과+9arXiv+9위키백과+9](https://arxiv.org/abs/2504.12285?utm_source=chatgpt.com)[Hugging Face+1](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T?utm_source=chatgpt.com)
    
    Hugging Face에 모델 가중치가 공개되어 있고, GitHub 저장소(bitnet.cpp)를 통해 효율적인 추론 프레임워크도 제공합니다


### 2. **BITNET**: 1980년대의 학술 컴퓨터 네트워크

- 1981년에 뉴욕시립대(CUNY)와 예일대(Yale University)에 의해 설립된 미국 중심의 대학 간 컴퓨터 네트워크입니다. 이름은 “Because It's There Network”에서 유래했지만, 나중엔 “Because It’s Time Network”라고도 불렸습니다.
    
- 주로 이메일, 파일 전송, LISTSERV 등을 통해 정보 교환을 지원했으며, 전 세계 여러 지역의 대학 네트워크에 연결되었습니다. 1990년대 인터넷의 확대와 함께 자연스럽게 사라져 갔습니다. [위키백과](https://en.wikipedia.org/wiki/BITNET?utm_source=chatgpt.com)


### 3. 기타 의미

- **BitNet (2017)**: Bit‑Regularized Deep Neural Networks  
    → 신경망의 파라미터 표현을 제한하는 정규화 전략으로, 학습 속도 향상 및 메모리 절약 효과를 보여준 모델입니다. MNIST, CIFAR‑10에서 실험됨. [위키백과+15arXiv+15arXiv+15](https://arxiv.org/abs/1708.04788?utm_source=chatgpt.com)
    
- **BitNet (2019)**: Bit‑Depth Expansion (이미지 관련 연구)  
    → 이미지의 비트 깊이를 학습 기반으로 확장(bit-depth expansion)하는 CNN 네트워크로, 사진의 자연스러운 색상 복구에 초점을 맞췄습니다. [arXiv+4arXiv+4위키백과+4](https://arxiv.org/abs/1910.04397?utm_source=chatgpt.com)

---

[ google notebookLM 요약 ]

**BitLinear 계층**:

    ◦ 기존의 `nn.Linear` 계층을 대체하는 드롭인(drop-in) 방식으로, **1-비트 가중치를 처음부터 학습**합니다.

    ◦ 가중치(`W`)는 **+1 또는 -1로 이진화**되며, 활성화(`x`)는 8-비트 정밀도로 양자화됩니다.

    ◦ 최적화 프로그램 상태와 기울기(gradients)는 학습 중 높은 정밀도(high precision)로 유지되어 학습 안정성을 확보합니다.

• **LayerNorm(SubLN) 통합**:

    ◦ 활성화 양자화 전에 LayerNorm 기능을 도입하여 출력의 분산(variance)을 보존하고 학습 안정성을 향상시킵니다. 이는 트랜스포머의 SubLN과 동일한 방식으로 구현됩니다.

• **그룹 양자화 및 정규화(Group Quantization and Normalization)**:

    ◦ 모델 병렬 처리(model parallelism)의 효율성을 높이기 위해 가중치와 활성화를 그룹으로 나누어 각 그룹의 매개변수를 독립적으로 추정합니다. 이를 통해 추가적인 통신 없이 모델 병렬 처리가 가능해집니다.

• **스크래치로부터 학습(Training from Scratch)**:

    ◦ 대부분의 기존 양자화 방법이 학습 후 양자화(Post-Training Quantization)인 반면, BitNet은 **양자화-인식 학습(Quantization-Aware Training)** 방식을 사용하여 모델이 양자화된 표현에 최적화되도록 처음부터 학습됩니다. 이는 낮은 정밀도에서도 상당한 정확도 손실을 방지합니다.

    ◦ 비차등 함수(non-differentiable functions)를 우회하여 기울기를 근사하는 **Straight-through estimator (STE)**를 사용합니다.

    ◦ 혼합 정밀도 학습(Mixed precision training)을 통해 가중치와 활성화는 낮은 정밀도로, 기울기와 최적화 프로그램 상태는 높은 정밀도로 유지됩니다.

• **높은 학습률(Large Learning Rate) 활용**:

    ◦ BitNet은 큰 학습률에서도 수렴할 수 있으며, 이는 FP16 트랜스포머가 발산하는 것과 대조적입니다. 이러한 최적화 이점을 통해 더 나은 수렴을 달성할 수 있습니다.

**성능 및 효율성 측면에서 BitNet의 이점**:

• **메모리 사용량 및 에너지 소비 대폭 감소**: FP16 트랜스포머 및 최첨단 8-비트 양자화 방법과 비교하여 메모리 사용량과 에너지 소비를 **상당히 줄여줍니다**. 특히 곱셈 연산에서 큰 에너지 절약을 제공합니다.

• **경쟁력 있는 성능**: 언어 모델링에서 경쟁력 있는 성능을 달성하며, 다운스트림 작업 정확도에서도 우수한 결과를 보여줍니다.

• **확장 법칙(Scaling Law) 준수**: 풀-정밀도 트랜스포머와 유사한 확장 법칙을 따르며, 이는 BitNet이 더 큰 LLM으로도 효과적으로 확장될 수 있음을 시사합니다. 고정된 계산 예산 내에서 훨씬 더 나은 손실을 달성하며, 동일한 성능을 얻는 데 필요한 추론 비용이 훨씬 적습니다.

• **기존 양자화 방법 능가**: 1-비트 모델임에도 불구하고 최첨단 학습 후 양자화 방법(Absmax, SmoothQuant, GPTQ, QuIP 등)보다 **일관되게 우수한 성능**을 보여줍니다.

결론적으로, BitNet은 1-비트 트랜스포머 아키텍처를 통해 대규모 언어 모델의 **메모리 및 계산 효율성을 크게 향상시키면서도 경쟁력 있는 성능을 유지**하는 혁신적인 접근 방식을 제시합니다

---
[ git_hub 모델 설명 ]

BitNet은 대규모 언어 모델(LLMs)을 위한 확장 가능하고 안정적인 1-비트 트랜스포머 아키텍처입니다. 이 모델의 설계는 기존 트랜스포머와 유사한 구조를 채택하면서도 효율성을 극대화하는 데 중점을 둡니다.

**Transformer와 동일한 레이아웃**: BitNet은 **트랜스포머와 동일한 레이아웃**을 사용합니다. 이는 **셀프-어텐션(self-attention) 및 피드-포워드 네트워크(feed-forward networks) 블록을 쌓아 올린 구조**로 구성됩니다.

**핵심적인 차이점: BitLinear 계층**: 바닐라(기존) 트랜스포머와 비교했을 때, BitNet은 기존의 행렬 곱셈 대신 **1-비트 모델 가중치를 사용하는 BitLinear 계층(수식 11)**을 사용합니다. `BitLinear`는 `nn.Linear` 계층을 대체하는 드롭인(drop-in) 방식으로, **1-비트 가중치를 처음부터 학습**하도록 설계되었습니다.

**정밀도 유지**: 가중치는 1-비트로 이진화되고 활성화는 8-비트 정밀도로 양자화되지만, BitNet은 다른 구성 요소들의 정밀도는 높게 유지합니다. 이러한 구성 요소들은 다음과 같습니다:

• **잔여 연결(residual connections) 및 계층 정규화(layer normalization)**: 이들은 대규모 언어 모델에서 계산 비용이 무시할 수 있을 정도로 작기 때문에 높은 정밀도를 유지합니다.

• **QKV 변환(QKV transformation)**: 모델 크기가 커질수록 QKV 변환의 계산 비용이 매개변수 투영(parametric projection)보다 훨씬 작아지므로 높은 정밀도를 유지합니다.

• **입력/출력 임베딩**: 언어 모델이 샘플링을 수행하기 위해 높은 정밀도의 확률을 사용해야 하므로, 입력/출력 임베딩의 정밀도를 보존합니다.

이러한 설계를 통해 BitNet은 언어 모델링에서 **경쟁력 있는 성능**을 달성하면서도, FP16 트랜스포머 및 최첨단 8-비트 양자화 방법과 비교하여 **메모리 사용량과 에너지 소비를 대폭 절감**합니다. 또한, BitNet은 풀-정밀도 트랜스포머와 유사한 **확장 법칙(scaling law)을 따르며**, 이는 BitNet이 효율성과 성능 이점을 유지하면서 더 큰 언어 모델로 효과적으로 확장될 수 있음을 시사합니다

---
### 여기서 잠깐! 트랜스포머란?

- **AI 모델 아키텍처**(설계도) 이름
    
- 2017년 Google의 _"Attention is All You Need"_ 논문에서 제안
    
- 자연어 처리(NLP)에서 GPT, BERT 같은 대형 언어 모델의 기본 뼈대
    
- 주요 구성:
    
    1. **Self-Attention**: 입력 문장 내 단어들이 서로 어떤 관련이 있는지 계산
        
    2. **Feed-Forward Network (FFN)**: Attention 결과를 더 복잡하게 변환
        
    3. **Residual Connection + LayerNorm**: 안정적 학습을 위한 보조 장치  
        → 이런 블록을 여러 층 쌓아 올린 구조

RNN -> RNN + attention -> 트랜스포머(self-attention)

트랜스포머 모델의 핵심 기능은 셀프 어텐션 [메커니즘이](https://www.ibm.com/kr-ko/think/topics/attention-mechanism)며, 이를 통해 입력 시퀀스의 각 구성 요소 간의 관계(또는 종속성)를 감지하는 놀라운 능력을 얻습니다. 기존 RNN 및 CNN 아키텍처와 달리, 트랜스포머는 오직 어텐션 레이어와 표준 피드포워드 레이어만 사용합니다.

셀프 어텐션의 이점, 특히 트랜스포머 모델이 이를 계산하기 위해 사용하는 멀티헤드 어텐션 기법은 트랜스포머가 이전에 최첨단이었던 RNN 및 CNN의 성능을 능가할 수 있도록 합니다.

`트랜스포머 모델이 등장하기 전 대부분의 NLP 작업은 RNN에 의존했습니다. RNN이 순차 데이터를 처리하는 방식은 본질적으로 _직렬화되어_ 있으며, 입력 시퀀스의 요소를 한 번에 하나씩 특정 순서로 수집합니다.  
  
`이는 RNN이 장거리 종속성을 캡처하는 능력을 저해하므로 RNN은 짧은 텍스트 시퀀스만 효과적으로 처리할 수 있습니다.  
`이러한 결함은 [장단기 메모리 네트워크(LSTM)](https://video.ibm.com/recorded/131507960)의 도입으로 어느 정도 해결되었지만, 여전히 RNN의 근본적인 단점으로 남아 있습니다.

반대로 [어텐션 메커니즘은](https://www.ibm.com/kr-ko/think/topics/attention-mechanism) 전체 시퀀스를 동시에 검토하고 해당 시퀀스의 특정 시간 단계에 집중하는 방법과 시기를 결정할 수 있습니다.  
  
트랜스포머의 이러한 특성은 장기적인 종속성 이해 능력을 크게 향상시킬 뿐 아니라, 일련의 단계가 아닌 여러 계산 단계를 동시에 수행할 수 있는 _병렬화_를 가능하게 합니다.

병렬 처리에 적합하다는 점에서 트랜스포머 모델은 학습과 추론 모두에서 [GPU](https://www.ibm.com/kr-ko/topics/gpu)가 제공하는 성능과 속도를 최대한 활용할 수 있습니다. 이러한 가능성은 [자기 지도 학습](https://www.ibm.com/kr-ko/think/topics/self-supervised-learning)을 통해 전례 없이 방대한 데이터 세트에 대해 트랜스포머 모델을 학습시킬 수 있는 기회를 열어주었습니다.

특히 시각적 데이터의 경우, 트랜스포머는 합성곱 신경망(CNN)보다 몇 가지 이점을 제공합니다. CNN은 본질적으로 지역적이며, 입력 데이터를 작은 단위로 나누어 하나씩 처리하기 위해 [합성곱](https://developer.ibm.com/articles/introduction-to-convolutional-neural-networks/)을 사용합니다  
  
따라서 CNN 역시 서로 인접하지 않은 단어(텍스트 내) 또는 픽셀(이미지 내) 간의 상관관계와 같은 장기적인 종속성을 식별하는 데 어려움을 겪습니다. 어텐션 메커니즘은 이러한 제약을 받지 않습니다.

---
### BitNet 을 Python 에서 사용 가능 여부 및 참고 내용

## 1. Python은 단순 **바인딩(wrapper)** 역할

- `bitnet.cpp`의 핵심 연산은 **C++로 작성**되어 있고, CPU 최적화(1.58-bit 행렬 연산 등)도 C++ 쪽에서 수행됩니다.
    
- Python 스크립트(`run_inference.py` 등)는 이 C++ 코드를 불러와 모델 로드, 프롬프트 입력, 출력 처리 등을 **편하게 실행**하게 해주는 인터페이스입니다.
    
- 즉, Python이 **실행 환경**이지만, 성능과 효율성은 여전히 C++ 백엔드에서 나옵니다.
    
## 2. “Python으로는 BitNet 장점을 못쓴다”는 말의 의미

이건 보통 두 가지 경우를 가리킵니다.

1. **순수 Python + PyTorch 추론**
    
    - BitNet 모델을 그냥 `transformers` + `torch`로 로드해서 실행하면  
        1.58-bit 연산 대신 **float32 또는 int8 연산**으로 돌아가기 때문에,  
        CPU/GPU 메모리 절감과 속도 향상 같은 BitNet의 **핵심 장점**을 못 누립니다.
        
    - 예: Hugging Face에서 바로 `AutoModelForCausalLM.from_pretrained()` 하면 일반 양자화 모델과 비슷하게 동작.
        
2. **GPU에서 실행**
    
    - 현재 BitNet의 초저비트 연산 커널은 CPU용 C++ 구현이 중심입니다.  
        GPU에서 돌리면 일반 half/8bit 연산으로 변환되어 실행돼서 BitNet 전용 최적화 이득이 거의 없습니다.


## Tensorflow 에서의 C++ 코드 실행과의 차이점
#### 차이점

| 항목     | TensorFlow         | BitNet (bitnet.cpp 기반) |
| ------ | ------------------ | ---------------------- |
| 구조     | Python → C++ 커널 호출 | Python → C++ 추론 엔진 호출  |
| 커널 로딩  | lazy load (필요시 로드) | 실행 시 모델과 커널 직접 로드      |
| 목적     | 다양한 ML 연산 지원       | 초저비트 LLM 추론 전용         |
| GPU 지원 | 강력, CUDA 커널 포함     | 현재 CPU 최적화 중심          |

---
### BitNet 사용방법

1.  필요 환경 세팅
2. 소스 코드 다운로드
```bash
git clone --recursive https://github.com/microsoft/BitNet.git
cd BitNet
```
3. 의존성 설치 (2025.08.13 기준 python=3.10 도 설치 과정에선 문제없음)
```bash
conda create -n bitnet-cpp python=3.9 
conda activate bitnet-cpp
pip install -r requirements.txt
```
4. 모델 다운로드 및 환경 설정 (사전 학습된 BitNet 가중치 파일 다운)
```bash
# Manually download the model and run with local path
huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir models/BitNet-b1.58-2B-4T

python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s
	(bitnet.cpp 프로젝트에서 제공하는 환경 준비 도구)
```
cmake 가 없으면 추가로 설치해줘야 된다. (콘다 환경에서만 사용하려면 conda 로 설치)
`github 에서는 3.22 버전 이상을 요구
```bash
conda install anaconda::cmake
```
clang 이 없다면 마찬가지로 추가로 설치해줘야 된다.
`github 에선 18 버전 이상을 요구
`visual studio 를 통해서 설치할 것을 요구
https://github.com/microsoft/BitNet

`(visual studio 설치 후 visual studio 에서 별도 파일 설치란을 통해 clang 설치)
`(이후 visual studio 설치시 내려받은 pompt 에서 버전 확인 가능)
![[Pasted image 20250813212720.png]]

그러나 현재 작업 중인 ubuntu 의 anaconda 에서는 clang -v 확인되지 않을 수 있다.

이 내용 또한 git hub 의 하단 부에 가면 설명되어 있어있고, 다음 명령어를 실행하면 된다.
`커뮤니티로 내려 받았을 경우 Professional 에서 Comunity 로 바꿔줘야 된다.`
```bash
"C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat" -startdir=none -arch=x64 -host_arch=x64
```
위는 Windows 에서 visual studio bat 에 접근하는 방식으로 Linux 에서는 사용할 수 없다
`다음과 같이 conda 로 설치하는 방법이 있다. (gpt5 솔루션)`
```bash
conda install -c conda-forge cmake ninja clangxx_linux-64
```

설치하고 나면 다음과 같이 결과를 얻을 수 있다
![[Pasted image 20250813214618.png]]

이후 다시 아래 명령을 실행해보면
```bash
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s
```
다음과 같이 나오면 완료 된 것이다
![[Pasted image 20250813215103.png]]
```
INFO:root:Compiling the code using CMake.
```
`: CMake를 이용해서 BitNet C++ 추론 엔진 코드를 빌드 중
` 앞서 설치한 `cmake`, `clang`, `ninja` 등을 사용해 `build/` 디렉터리에 실행 파일과 라이브러리를 생성`

```
INFO:root:Loading model from directory models/BitNet-b1.58-2B-4T.
```
`: `-md` 옵션에서 지정한 모델 디렉터리를 불러오는 중
` 여기서는 `models/BitNet-b1.58-2B-4T` 경로에 다운로드된 BitNet 모델 파일(.gguf, tokenizer 등)을 로드`

```
INFO:root:GGUF model already exists at models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
```
`: `-q i2_s`로 지정한 양자화 버전(i2_s) 모델이 이미 존재하므로 재변환(quantization) 과정을 건 너뜀을 의미
` 즉, 새로 양자화하지 않고 바로 빌드된 엔진이 해당 gguf 모델을 사용할 수 있게 연결`

간혹 ![[Pasted image 20250813215934.png]]
다음과 같이 에러가 발생할 경우 complie.log 내용을 확인 해서 관련 에러라면 이 또한 github 에서 다루고 있으니 확인해보면 좋을 것 같다
```bash
# Run inference with the quantized model
python run_inference.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -p "You are a helpful assistant" -cnv
```
`python run_inference.py`
- BitNet 패키지 안에 있는 **추론 실행 스크립트**를 Python으로 실행
    
- 이 스크립트는 내부에서 **C++로 빌드된 bitnet.cpp 엔진**을 불러와서,
    
    - 모델 로드
    - 프롬프트 전달
    - 토큰 생성
    - 출력 표시  
        
	를 담당.

 `-m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf`
- **`-m` = model path**  
    로컬에 저장된 모델 파일(.gguf)을 지정합니다.
    
- 여기서 `.gguf` 파일은 1.58-bit 양자화된 BitNet 모델 가중치
    
- `i2_s`는 `setup_env.py -q i2_s`로 지정했던 **양자화 프리셋**을 의미
    
    - `i2` → integer 2-bit quantization
    - `_s` → small/compact 버전

`-p "You are a helpful assistant"`
- **`-p` = prompt**  
    모델에 처음 전달할 입력 문장
    
- 이 경우 프롬프트는 `"You are a helpful assistant"`로,  
    LLM이 이걸 읽고 그에 맞게 대화를 이어가도록 유도

 `-cnv`
- **`-c` = chat mode** + **`-nv` = no verbose**를 합쳐놓은 축약 형태일 가능성이 큼  
    (BitNet/llama.cpp 계열에서 종종 이렇게 붙임)
    
- chat mode → 모델이 채팅 대화 형식으로 동작
    
- no verbose → 불필요한 디버그 로그를 줄이고 출력만 표시
    
- 실제 의미는 `run_inference.py` 내부 `argparse` 부분에서 확인 가능하지만,  
    일반적으로 `-cnv`는 **대화 모드에서 깔끔하게 출력**하라는 옵션

실행 흐름
1. `run_inference.py`가 `bitnet.cpp` 엔진을 로드
2. 지정한 `.gguf` 모델 파일을 메모리에 적재
3. `"You are a helpful assistant"` 프롬프트를 토크나이즈 → C++ 엔진에 전달
4. CPU에서 1.58-bit 연산으로 토큰 생성
5. 생성된 응답을 콘솔에 출력

---
#### 첫 번째 테스트
![[화면 캡처 2025-08-14 084331.png]]
대답을 하다가 만다...

토큰 길이가 부족해서라고 GPT 는 대답을 해주긴 하는데..

토큰을 임의로 늘이던가 BitNet 에 맞는 질문 형식을 하라고 한다...
```bash
python run_inference.py \
  -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  -p "비둘기, 참새, 나비, 제비, 뻐꾸기 중 다른 하나를 한 단어로만 답하시오." \
  -cnv \
  --n-predict 256
```
실행시 `--n-predict 256` 으로 토큰 길이를 늘이는 방법을 사용하던가

```
BitNet은 2B~3B 소형 모델 + 양자화이므로,

- 명령은 짧고 명확하게
- 필요하면 출력 형식까지 지시
```
방식으로 지시하라고 한다.

#### 두 번째 테스트

실행 명령은 이전과 동일한 상황에서 토큰 값만 늘려주고 테스트 해봤다
![[Pasted image 20250814090913.png]]
너무 무섭다, 같은 말을 반복한다
![[Pasted image 20250814090820.png]]
다르게 질문도 해봤지만 그저 무서울 뿐이다.
![[화면 캡처 2025-08-14 092304.png]]

