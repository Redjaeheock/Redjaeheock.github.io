# Low Rank Adaptation of Large Language Models

[Arix]()
An important paradigm of natural language processing consists of large-scale pretraining on general domain data and adaptation to particular tasks or domains. As we pre-train larger models, full fine-tuning, which retrains all model parameters, becomes less feasible. Using GPT-3 175B as an example – deploying independent instances of fine-tuned models, each with 175B parameters, is prohibitively expensive. We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times. LoRA performs on-par or better than finetuning in model quality on RoBERTa, DeBERTa, GPT-2, and GPT-3, despite having fewer trainable parameters, a higher training throughput, and, unlike adapters, no additional inference latency. We also provide an empirical investigation into rank-deficiency in language model adaptation, which sheds light on the efficacy of LoRA. We release a package that facilitates the integration of LoRA with PyTorch models and provide our implementations and model checkpoints for RoBERTa, DeBERTa, and GPT-2 at

### \[GPT 번역]

자연어 처리(NLP)에서 중요한 패러다임 중 하나는 **일반 도메인 데이터에 대규모 사전 학습(pretraining)** 을 하고, 이후 특정 태스크나 도메인에 맞게 **적응(adaptation)** 하는 것입니다.  
하지만 모델이 커질수록, **모델 전체 파라미터를 다시 학습시키는 전체 파인튜닝(full fine-tuning)** 은 점점 비현실적이 됩니다.

예를 들어 GPT-3 175B 같은 초대형 모델을 생각해보면, 태스크마다 독립적인 파인튜닝 버전을 만들어 배포한다면 각 모델이 1,750억 개의 파라미터를 가지고 있어 비용이 너무 비싸집니다.

이를 해결하기 위해 우리는 **LoRA (Low-Rank Adaptation)** 를 제안합니다.  
LoRA는 사전 학습된 모델의 가중치는 고정(freeze)하고, **트랜스포머 아키텍처의 각 레이어에 학습 가능한 저차원(rank-decomposition) 행렬을 주입**하여, 다운스트림 태스크에 필요한 학습 파라미터 수를 크게 줄입니다.

- GPT-3 175B를 Adam 옵티마이저로 풀 파인튜닝했을 때와 비교하면,
    
    - LoRA는 **학습해야 할 파라미터 수를 10,000배 줄이고**,
        
    - **GPU 메모리 요구량도 3배 줄일 수 있습니다.**
        

그럼에도 불구하고 LoRA는 RoBERTa, DeBERTa, GPT-2, GPT-3 같은 모델에서 **성능이 풀 파인튜닝과 동등하거나 더 좋게 나왔습니다.**  
게다가 학습 속도(throughput)도 더 높고, 어댑터(adapters) 방식과 달리 **추론 속도(inference latency)가 늘어나지 않습니다.**

또한 우리는 **언어 모델 적응에서의 랭크 결핍(rank-deficiency)** 문제에 대한 실험적 분석을 제공하여, LoRA가 효과적인 이유를 더 잘 설명합니다.

마지막으로 PyTorch 모델에 LoRA를 쉽게 통합할 수 있는 패키지를 제공하고, RoBERTa, DeBERTa, GPT-2용 구현 및 모델 체크포인트를 함께 공개합니다.

### \[ 요약 ]

기존의 **full fine-tuning** 방식에서는 거대한 사전학습 가중치를 모두 다시 학습해야 했기 때문에,  
모델을 응용할 때마다 **전체 가중치를 그대로 복제·저장해야 하는 부담**이 존재했다.

그러나 **LoRA 아키텍처**는 대형 가중치를 **고정(freeze)** 한 채,  
그 변화량을 **저차원(저랭크) 행렬로 분해하여 학습**함으로써,  
**필요한 추가 파라미터만 활용할 수 있는 구조**를 만들었다.

이를 통해 LoRA는 **대형 언어 모델(LLM)의 경량화와 효율적 확장성**에 크게 이바지했다.


