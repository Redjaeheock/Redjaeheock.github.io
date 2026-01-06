
## Abstract


##### 원문
   State-of-the-art computer vision systems are trained to predict a fixed set of predetermined object categories. 
   This restricted form of supervision limits their generality and usability since additional labeled data is needed to specify any other visual concept. 
   Learning directly from raw text about images is a promising alternative which leverages a much broader source of supervision. 

   We demonstrate that the simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400 million (image, text) pairs collected from the internet. After 
   pre-training, natural language is used to reference learned visual concepts (or describe new ones) enabling zero-shot transfer of the model to downstream tasks. 

   We study the performance of this approach by benchmarking on over 30 different existing computer vision datasets, spanning tasks such as OCR, action recognition in videos, 
   geo-localization, and many types of fine-grained object classification. 
   The model transfers non-trivially to most tasks and is often competitive with a fully supervised baseline without the need for any dataset specific training. 
   For instance, we match the accuracy of the original ResNet-50 on ImageNet zero-shot without needing to use any of the 1.28 million training examples it was trained on. 

   We release our code and pre-trained model weights at https://github.com/OpenAI/CLIP.

##### 번역
   _최신(state-of-the-art) 컴퓨터 비전 시스템들은 미리 정해진 고정된 객체 범주 집합을 
   예측하도록 학습된다. 
   이러한 제한된 형태의 지도(supervision)는, 다른 시각적 개념을 지정하려면 추가적인 라벨링된 데이터가 필요하기 때문에, 그 일반성과 사용성을 제한한다. 
   이미지에 대한 원시(raw) 텍스트로부터 직접 학습하는 것은 훨씬 더 폭넓은 형태의 지도를 활용할 수 있는 유망한 대안이다.

   _우리는 어떤 캡션이 어떤 이미지와 짝을 이루는지를 예측하는 단순한 사전 학습(pre-training) 작업이, 인터넷에서 수집한 4억 개의 (이미지, 텍스트) 쌍으로 구성된 데이터셋 위에서, 처음부터(from scrat_ch) 최첨단 수준의 이미지 표현을 학습하는 효율적이고 확장 가능한 방법임을 보인다.
   사전 학습 이후에는, 자연어(natural language)가 학습된 시각적 개념을 참조하거나(reference) 새로운 시각적 개념을 설명(describe)하는 데 사용될 수 있으며, 이를 통해 모델을 다운스트림 작업에 제로샷(zero-shot) 방식으로 전이(transfer)할 수 있다.

   _우리는 OCR, 비디오에서의 행동 인식(action recognition in videos), 지리적 위치 추정
   (geo-localization), 그리고 여러 종류의 세분화된(fine-grained) 객체 분류를 포함하는 30개가 넘는 기존 컴퓨터 비전 데이터셋들에서, 이러한 접근법의 성능을 벤치마킹하여 연구하였다. 
   이 모델은 대부분의 작업에 대해 단순하지 않은(non-trivial) 방식으로 전이되며, 어떤 데이터셋에 대해서도 별도의 데이터셋 특화 학습(dataset specific training) 없이도, 완전 지도(fully supervised) 방식의 기준(baseline)과 종종 경쟁력 있는 성능을 보인다.
   예를 들어, 우리는 원래의 ResNet-50이 ImageNet에서 달성한 정확도를, 그 모델이 학습할 때 사용했던 128만(1.28 million) 개의 ImageNet 학습 예제들 중 어느 것도 사용하지 않고도, ImageNet에 대해 제로샷(zeroshot)으로 동일한(match) 정확도를 달성한다.

   _코드와 사전 학습된 모델 가중치는 다음에서 공개한다: [https://github.com/OpenAI/CLIP](https://github.com/OpenAI/CLIP)_

#### 요약:
CLIP은 기존 이미지 인식 모델들이 **‘미리 지정된 범주(label)에 기반한 학습’** 에 묶여 있어  
이미지의 다양한 시각적 해석을 제한받는 문제를 개선하고자,  
인터넷에서 수집한 대규모 **이미지–캡션 쌍**을 활용한 학습 방식을 도입하였다.  
그 결과, **OCR, 비디오 행동 인식, 지리적 위치 추정**과 같이 기존에는 **작업별로  
별도의 특화 학습이 필요했던** 문제들에서도,  
**추가 학습 없이 기존 특화 모델들과 유사한 수준의 해석 능력**을 보여주는 것이 확인되었다.

---

## 1. Introduction and Motivating Work

##### 원문
   Pre-training methods which learn directly from raw text have revolutionized NLP over the last few years (Dai & Le, 2015; Peters et al., 2018; Howard & Ruder, 2018; Rad- ford et al., 2018; Devlin et al., 2018; Raffel et al., 2019).
   Task-agnostic objectives such as autoregressive and masked language modeling have scaled across many orders of magnitude in compute, model capacity, and data, steadily improving capabilities.
   The development of “text-to-text” as a standardized input-output interface (McCann et al., 2018; Radford et al., 2019; Raffel et al., 2019) has enabled task-agnostic architectures to zero-shot transfer to downstream datasets removing the need for specialized output heads or dataset specific customization. 
   Flagship systems like GPT-3 (Brown et al., 2020) are now competitive across many tasks with bespoke models while requiring little to no dataset specific training data.

##### 번역
   ***원시(raw) 텍스트로부터 직접 학습하는 사전 학습(pre-training) 기법**은 최근 수년간 NLP 분야에서 혁명적인 발전을 이끌어왔다 (Dai & Le, 2015; Peters et al., 2018; Howard & Ruder, 2018; Radford et al., 2018; Devlin et al., 2018; Raffel et al., 2019).
   **자동회귀(autoregressive)** 또는 **마스크드 언어 모델링(masked language modeling)** 과 같은 **작업 비종속(task-agnostic) 목적 함수**들은 연산량(compute), 모델 용량(capacity), 
   그리고 데이터 규모에 걸쳐 수십 배로 확장됨에 따라 모델의 능력을 꾸준히 향상시켜 왔다.
   **입·출력을 ‘text-to-text’로 표준화한 인터페이스(text-to-text interface)의 개발** (McCann et al., 2018; Radford et al., 2019; Raffel et al., 2019)은 작업 비특화(task-agnostic) 아키텍처가 
   다운스트림 데이터셋에 대해 별도의 출력 헤드(output heads)나 데이터셋 특화(customization) 없이 제로샷(zeroshot) 전이를 수행할 수 있도록 만들었다.
   GPT-3(Brown et al., 2020)와 같은 대표적(flagship) 시스템들은 이제 거의 데이터셋 특화 학습 없이도 여러 작업에서 특수 목적(bespoke) 모델들과 경쟁 가능한 수준에 도달하였다.

##### 원문
   These results suggest that the aggregate supervision accessible to modern pre-training methods within web-scale collections of text surpasses that of high-quality crowd-labeled NLP datasets.
   However, in other fields such as computer vision it is still standard practice to pre-train models on crowd-labeled datasets such as ImageNet (Deng et al., 2009).
   Could scalable pre-training methods which learn directly from web text result in a similar breakthrough in computer vision? Prior work is encouraging. 

##### 번역
   _이러한 결과들은, 웹 규모(web-scale)의 텍스트 컬렉션에 존재하는 **현대 사전 학습(pre-training) 기법들이 접근할 수 있는 전체적(supervision) 지도 신호의 양**이, **고품질 크라우드 
   라벨링된 NLP 데이터셋에서 얻을 수 있는 지도 신호를 능가**한다는 사실을 시사한다. 
   그러나 컴퓨터 비전과 같은 다른 분야에서는 여전히 ImageNet(Deng et al., 2009)과 같은 
   **크라우드 라벨링된 데이터셋으로 모델을 사전 학습**하는 것이 **표준 관행**이다.
   **웹 텍스트로부터 직접 학습하는 확장 가능한 사전 학습 방법이 컴퓨터 비전에서도 이와 유사한 돌파구(breakthrough)를 만들어낼 수 있을까?** 이전 연구는 그 가능성을 보여준다.

##### 원문
   Over 20 years ago Mori et al. (1999) explored improving content based image retrieval by training a model to predict the nouns and adjectives in text documents paired with images.
   Quattoni et al. (2007) demonstrated it was possible to learn more data efficient image representations via manifold learning in the weight space of classifiers trained to predict words in captions associated with images. 
   Srivastava & Salakhutdinov (2012) explored deep representation learning by training multimodal Deep Boltzmann Machines on top of low-level image and text tag features. 
   Joulin et al. (2016) modernized this line of work and demonstrated that CNNs trained to predict words in image captions learn useful image representations. 
   They converted the title, description, and hashtag metadata of images in the YFCC100M dataset (Thomee et al., 2016) into a bag-of-words multi-label classification task and showed that pretraining AlexNet (Krizhevsky et al., 2012) to predict these labels learned representations which preformed similarly to ImageNet-based pre-training on transfer tasks. Li et al. (2017) then extended this approach to predicting phrase ngrams in addition to individual words and demonstrated the ability of their system to zero-shot transfer to other image classification datasets by scoring target classes based on their dictionary of learned visual n-grams and predicting the one with the highest score. Adopting more recent architectures and pre-training approaches, VirTex (Desai & Johnson, 2020), ICMLM (Bulent Sariyildiz et al., 2020), and ConVIRT (Zhang et al., 2020) have recently demonstrated the potential of transformer-based language modeling, masked language modeling, and contrastive objectives to learn image representations from text

#### 번역
   _20여 년 전 Mori et al. (1999)은 이미지와 짝지어진 텍스트 문서에 포함된 명사와 형용사를 
   예측하도록 모델을 학습함으로써 **콘텐츠 기반 이미지 검색(content-based image retrieval)** 을 개선하는 방식을 탐구했다.
   Quattoni et al. (2007)은 이미지와 연관된 캡션 속 단어들을 예측하도록 학습된 분류기의  
   가중치 공간(weight space)에서 **매니폴드 학습(manifold learning)** 을 수행하여 **더 데이터 효율적인 이미지 표현을 학습할 수 있음**을 보였다.
   Srivastava & Salakhutdinov (2012)는 저수준 이미지 특징과 텍스트 태그 특징 위에 멀티모달 Deep Boltzmann Machine을 쌓아 **깊은 표현 학습(deep representation learning)** 을 탐구했다.
   Joulin et al. (2016)은 이 연구 흐름을 현대화하여, **이미지 캡션 내 단어들을 예측**하도록 CNN을 학습시키면 **유용한 이미지 표현을 학습**할 수 있다는 것을 입증했다.
   그들은 YFCC100M 데이터셋(Thomee et al., 2016)의 사진 제목(title), 설명(description),
   해시태그(hashtag) 메타데이터를 **bag-of-words 기반의 다중 라벨 분류(multi-label classification) 작업**으로 변환했고, AlexNet(Krizhevsky et al., 2012)을 이러한 라벨을 예측하도록 사전 학습시켜 **전이 학습(transfer tasks)** 에서 **ImageNet 기반 사전학습과 유사한 성능**을 
   확보했다.
   Li et al. (2017)은 이 접근 방식을 확장하여 개별 단어뿐 아니라 **구문 단위(phrase n-grams)** 도 예측하게 했고, 그들의 시스템이 학습한 **시각적 n-gram 사전을 기반**으로 타깃 클래스와의 매칭 점수를 계산하여 가장 높은 점수를 예측함으로써 **다른 이미지 분류 데이터셋에 제로샷 전이(zero-shot transfer)** 할 수 있음을 보였다.
   보다 최신의 아키텍처와 사전 학습 기법을 채택한 VirTex(Desai & Johnson, 2020), ICMLM(Bulent Sariyildiz et al., 2020), 그리고 ConVIRT(Zhang et al., 2020)는 **트랜스포머 
   기반 언어 모델링, 마스크드 언어 모델링**, 그리고 **대조학습(contrastive objectives)이** 텍스트로부터 이미지 표현을 학습하는 데에 잠재력이 있음을 최근에 보여주었다.

#### 원문
   While exciting as proofs of concept, using natural language supervision for image representation learning is still rare.
   This is likely because demonstrated performance on common benchmarks is much lower than alternative approaches.
   For example, Li et al. (2017) reach only 11.5% accuracy on ImageNet in a zero-shot setting. This is well below the 88.4% accuracy of the current state of the art (Xie et al., 2020). 
   It is even below the 50% accuracy of classic computer vision approaches (Deng et al., 2012). Instead, more narrowly scoped but well-targeted uses of weak supervision have improved performance. 
   Mahajan et al. (2018) showed that predicting ImageNet-related hashtags on Instagram images is an effective pre-training task. When fine-tuned to ImageNet these pre-trained models increased accuracy by over 5% and improved the overall state of the art at the time.
   Kolesnikov et al. (2019) and Dosovitskiy et al. (2020) have also demonstrated large gains on a broader set of transfer benchmarks by pre-training models to predict the classes of the noisily labeled JFT-300M dataset.

##### 번역
   _개념 증명(proofs of concept)으로서는 흥미롭지만, 이미지 표현 학습(image representation learning)을 위해 **자연어 지도를 활용하는 것은 여전히 드물다.** 
   이는, 일반적인 벤치마크에서 입증된 성능이 다른 접근법들에 비해 훨씬 낮기 때문인 것으로 
   보인다.
   예를 들어, Li et al. (2017)은 **ImageNet 제로샷(zeroshot) 설정에서 단 11.5% 정확도**를 기록했다.
   이는 최신(state-of-the-art) 정확도인 88.4%(Xie et al., 2020)보다 훨씬 낮다.
   이는 고전적 컴퓨터 비전 방법(Deng et al., 2012)의 **50% 정확도보다도 낮다.**
   대신에, 더 범위가 좁지만 **특정 목적(targeted)** 으로 잘 설계된 **약지도(weak supervision)** 
   형태의 접근법들은 성능을 개선해왔다.
   Mahajan et al. (2018)은 **Instagram 이미지에서 ImageNet 관련 해시태그를 예측하는 것**이 
   효과적인 사전 학습 태스크임을 보여주었다.
   이렇게 사전 학습된 모델을 ImageNet에 파인튜닝하자 정확도가 **5% 이상 상승**했고, 당시 
   전반적인 SOTA 또한 개선되었다.
   Kolesnikov et al. (2019)과 Dosovitskiy et al. (2020) 또한 **노이즈가 포함된 라벨을 가진 
   JFT-300M 데이터셋**의 클래스를 예측하도록 모델을 사전 학습하여 더 넓은 전이 벤치마크에서 큰 성능 향상을 입증했다.

##### 원문
   This line of work represents the current pragmatic middle ground between learning from a limited amount of supervised “gold-labels” and learning from practically unlimited amounts of raw text.
   However, it is not without compromises.
   Both works carefully design, and in the process limit, their supervision to 1000 and 18291 classes respectively.
   Natural language is able to express, and therefore supervise, a much wider set of visual concepts through its generality.
   Both approaches also use static softmax classifiers to perform prediction and lack a mechanism for dynamic outputs.
   This severely curtails their flexibility and limits their “zero-shot” capabilities.

##### 번역
   _이 연구 흐름은, 제한된 양의 **지도(gold-label) 데이터**로 학습하는 방법과 사실상 무제한에
   가까운 **원시(raw) 텍스트**로 학습하는 방법 사이에서 현실적인 중간 지점을 나타낸다.
   그러나 이는 타협점을 포함한다.
   두 연구 모두 그들의 지도를 각각 1000개와 18,291개의 클래스로 제한하도록 신중히 설계했다.
   자연어는 그 일반성으로 인해 훨씬 더 넓은 범위의 시각적 개념을 표현할 수 있고 따라서 지도  할 수 있다. 
   두 접근법은 또한 예측을 수행하기 위해 정적 softmax 분류기를 사용하며 동적 출력 메커니즘을 결여한다. 
   이는 유연성을 크게 줄이며 “제로샷(zero-shot)” 능력을 제한한다.

##### 원문
   A crucial difference between these weakly supervised models and recent explorations of learning image representations directly from natural language is scale. 
   While Mahajan et al. (2018) and Kolesnikov et al. (2019) trained their models for accelerator years on millions to billions of images, VirTex, ICMLM, and ConVIRT trained for accelerator days on one to two hundred thousand images.
   In this work, we close this gap and study the behaviors of image classifiers trained with natural language supervision at large scale.
   Enabled by the large amounts of publicly available data of this form on the internet, we create a new dataset of 400 million (image, text) pairs and demonstrate that a simplified version of ConVIRT trained from scratch, which we call CLIP, for Contrastive Language-Image Pre-training, is an efficient method of learning from natural language supervision.
   We study the scalability of CLIP by training a series of eight models spanning almost 2 orders of magnitude of compute and observe that transfer performance is a smoothly predictable function of compute (Hestness et al., 2017; Kaplan et al., 2020).
   We find that CLIP, similar to the GPT family, learns to perform a wide set of tasks during pre-training including OCR, geo-localization, action recognition, and many others.
   We measure this by benchmarking the zero-shot transfer performance of CLIP on over 30 existing datasets and find it can be competitive with prior task-specific supervised models. We also confirm these findings with linear-probe representation learning analysis and show that CLIP outperforms the best publicly available ImageNet model while also being more computationally efficient. We additionally find that zero-shot CLIP models are much more robust than equivalent accuracy supervised ImageNet models which suggests that zero-shot evaluation of task-agnostic models is much more representative of a model’s capability. These results have significant policy and ethical implications, which we consider in Section 7.

##### 번역
   _**약지도(weak supervision)** 모델들과 자연어로부터 직접 이미지 표현을 학습하는 최근 탐구 사이의 중요한 차이점은 규모이다. 
   Mahajan et al. (2018)과 Kolesnikov et al. (2019)은 수백만에서 수십억 개의 이미지에 대해 
   가속기 기준 여러 해 동안 모델을 학습했다.
   VirTex, ICMLM, ConVIRT는 10만에서 20만 개의 이미지로 가속기 기준 며칠 동안 학습했다.
   이 연구에서는 이 격차를 해소하고 대규모 자연어 지도와 함께 학습된 이미지 분류기의 동작을 연구한다. 
   이 형식의 공개 데이터의 대규모 이용 가능성에 의해, 우리는 4억 개의 (이미지, 텍스트) 쌍으로 구성된 새로운 데이터셋을 만든다. 
   그리고 처음부터 학습된 ConVIRT의 단순화 버전을, 우리가 CLIP(Contrastive Language-Image Pre-training)이라 부르는 모델로 명명하고, 이것이 자연어 지도로부터 학습하는 효율적인 방법임을 보인다.
   우리는 계산량이 거의 두 자릿수 차이를 아우르는 여덟 개의 모델을 학습하여 CLIP의 확장성을 연구한다. 
   그리고 전이 성능이 계산량의 함수로서 부드럽고 예측 가능함을 관찰한다(Hestness et al., 2017; Kaplan et al., 2020). 
   우리는 CLIP이 GPT 계열과 유사하게, OCR, 지리적 위치 추정, 행동 인식 등을 포함한 넓은 범위의 작업을 사전 학습 중에 수행하는 법을 학습한다는 것을 발견한다.
   우리는 30개 이상의 기존 데이터셋에서 CLIP의 제로샷 전이 성능을 벤치마킹하여 이전 작업 
   특화 지도 모델들과 경쟁력이 있을 수 있음을 발견한다.
   우리는 또한 linear-probe 표현 학습 분석으로 이러한 결과를 확인하고, CLIP이 공개된 최고의 ImageNet 모델보다 뛰어나며 더 계산 효율적임을 보인다.
   우리는 또한 제로샷 CLIP 모델이 동일한 정확도의 지도 ImageNet 모델보다 훨씬 더 강건하다는 것을 발견한다. 
   이는 작업 비특화 모델의 제로샷 평가가 모델의 능력을 훨씬 더 잘 나타낸다는 것을 시사한다.
   이 결과들은 중요한 정책적·윤리적 함의를 가지며, 이는 7장에서 다룬다.

---
## 2. Approach

### 2.1. Natural Language Supervision
   At the core of our approach is the idea of learning perception from supervision contained in natural language. 
   Asdiscussed in the introduction, this is not at all a new idea, however terminology used to describe work in this space is varied, even seemingly contradictory, and stated motivations
   are diverse. 
   Zhang et al. (2020), Gomez et al. (2017), Joulin et al. (2016), and Desai & Johnson (2020) all introduce methods which learn visual representations from text paired with images but describe their approaches as unsupervised, self-supervised, weakly supervised, and supervised respectively.

   We emphasize that what is common across this line of work is not any of the details of the particular methods used but the appreciation of natural language as a training signal. 
   All these approaches are learning from natural language super-vision.
   Although early work wrestled with the complexity of natural language when using topic model and n-gram representations, improvements in deep contextual representation learning suggest we now have the tools to effectively leverage this abundant source of supervision (McCann et al., 2017).
   
   Learning from natural language has several potential strengths over other training methods. It’s much easier to scale natural language supervision compared to standard crowd-sourced labeling for image classification since it does not require annotations to be in a classic “machine learning compatible format” such as the canonical 1-of-N majority vote “gold label”. 
   Instead, methods which work on natural language can learn passively from the supervision contained in the vast amount of text on the internet. 
   Learning from natural language also has an important advantage over most unsupervised or self-supervised learning approaches in that it doesn’t “just” learn a representation but also connects that representation to language which enables flexible zero-shot transfer. In the following subsections, we detail the specific approach we settled on.

   ***2.1. 자연어/캡션 지도/감독(Natural Language Supervision)**
   
   _우리 접근법의 핵심에는 **자연어가 담고 있는 지도(supervision)로부터 지각(perception)을 학습한다는 아이디어**가 있다.  
   서론에서 논의했듯이, 이는 전혀 새로운 아이디어가 아니지만, 이 분야의 연구를 설명하는 데 사용되는 **용어는 매우 다양하거나 심지어 서로 모순적으로 보이기도 하며**, 제시되는 동기도 일관되지 않다.
   예를 들어,  Zhang et al. (2020), Gomez et al. (2017), Joulin et al. (2016), Desai & Johnson (2020)는 모두 **텍스트와 이미지가 짝지어진 데이터를 사용해 시각적 표현을 학습하는 방법**을 제안했지만, 각각의 접근 방식을 **비지도(unsupervised)**, **자기지도(self-supervised)**, 
   **약지도(weakly supervised)**, **지도(supervised)**라고 설명한다.

   _우리가 강조하는 바는, 이 연구 흐름에서 공통적인 것은 **특정 방법론의 세부사항이 아니라**, **자연어를 하나의 훈련 신호(training signal)로 활용한다는 점**이라는 것이다. 
   이 모든 접근법은 자연어 지도(natural language supervision)로부터 학습하고 있다.
   초기 연구는 토픽 모델(topic model)이나 n-gram 기반 표현을 사용할 때 자연어의 복잡성 때문에 어려움을 겪었지만, 딥 컨텍스추얼 표현 학습(deep contextual representation learning)의 
   발전 덕분에 이 풍부한 지도 신호를 효과적으로 활용할 수 있는 도구가 갖춰졌음을 시사한다 (McCann et al., 2017).

   _자연어로부터 학습하는 것은 다른 훈련 방법보다 잠재적인 여러 강점을 지닌다. 
   우선, 자연어 지도는 전통적인 이미지 분류용 크라우드소싱 라벨링에 비해 **확장성이 훨씬 높다**. 왜냐하면 “머신러닝 친화적 형식”, 예컨대 1-of-N 방식의 다수결 “골드 라벨(gold label)”처럼 고전적인 주석(annotation) 형태를 요구하지 않기 때문이다.
   대신, 자연어를 기반으로 작동하는 방법들은 인터넷에 존재하는 방대한 양의 텍스트에 포함된 지도 신호로부터 **수동적으로(passively)** 학습할 수 있다.
   또한, 자연어 학습은 대부분의 비지도·자기지도 학습 방식에 비해 중요한 장점을 갖는데, 
   그것은 이 방법이 **표현(representation) 자체만을 학습하는 데에 그치지 않고**, 그 표현이 **언어와 연결되도록 학습한다는 점**이다. 이 덕분에 **유연한 제로샷 전이(flexible zero-shot transfer)**가 가능해진다.
   이후의 하위 섹션에서는 우리가 최종적으로 선택한 구체적인 접근 방식에 대해 상세히 설명한다.
### 2.2. Creating a Sufficiently Large Dataset 
### 2.3. Selecting an Efficient Pre-Training Method 
### 2.4. Choosing and Scaling a Model 
### 2.5. Training

---
## 3. Experiments 
### 3.1. Zero-Shot Transfer 
#### 3.1.1. MOTIVATION 
#### 3.1.2. USING CLIP FOR ZERO-SHOT TRANSFER 
#### 3.1.3. INITIAL COMPARISON TO VISUAL N-GRAMS 
#### 3.1.4. PROMPT ENGINEERING AND ENSEMBLING 
#### 3.1.5. ANALYSIS OF ZERO-SHOT CLIP PERFORMANCE 
### 3.2. Representation Learning 
### 3.3. Robustness to Natural Distribution Shift 

---
## 4. Comparison to Human Performance 

---
## 5. Data Overlap Analysis 

---
## 6. Limitations 

---
## 7. Broader Impacts 

### 7.1. Bias 
### 7.2. Surveillance 
### 7.3. Future Work 

---
## 8. Related Work 

---
## 9. Conclusion 

---
## References 

---
## checkpoints

