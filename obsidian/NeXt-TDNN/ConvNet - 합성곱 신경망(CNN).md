# ConvNet: **Convolutional Neural Network**

합성곱 신경망은 합성곱 연산을 사용하는 신경망 중 하나로서, 주로 음성 인식이나 시각적 이미지를 분석하는데 사용된다. 

합성곱 신경망이 일반 신경망과 다른 점은 일반적인 신경망들이 이미지 데이터를 원본 그대로 1차원 신경망인 Fully-Connected layer(FC layer 혹은 Dense layer)에 입력되어 전체 특성을 학습하고 처리하는 데 반해, **합성곱 신경망은 FC layer를 거치기 전에 Convolution 및 Poolling과 같은 데이터의 주요 특징 벡터를 추출하는 과정을 거친 후 FC layer로 입력되게 된다.**  

그렇기 때문에 대부분의 이미지 인식 분야는 딥러닝 기반의 합성곱 신경망이 주를 이루고 있다.

---
## ConvNet 구조

![[Pasted image 20250926084709.png]]

합성곱 신경망은 기본적으로 위와 같은 구조로 이루어져있다.

크게 3 가지 영역으로 분류한다면
- Input Layer
- Convolutional layer ~ Max pooling layer
- Fully-connected layer ~ output layers

Convolutional Layer와 Pooling Layer라는 과정을 거치게 되는데 이는 Input Image의 주요 특징 벡터를 추출하는 과정이다

그 후 추출된 주요 특징 벡터들이 FC Layer를 거치며 1차원 벡터로 변환되고, 
마지막 Output layer에서 활성화 함수인 Softmax함수를 통해 각 해당 클래스의 확률로 출력된다

### Input Layer

- 입력된 이미지 데이터가 최초로 거치게 되는 Layer
- 입력된 이미지 데이터의 기본 형태
	- 높이
	- 넓이
	- 채널
		- 채널 크기 1: Gray Scale (흑백 이미지)
		- 채널 크기 3: RGB (컬러 이미지 - 가산혼합)
		- 채널 크기 4: RGBA (컬러 + 투명도(Alpha) 이미지)
		- 채널 크기 4: CMYK (\[사이언, 마젠타, 옐로우, 검정]이미지 - 감산혼합)
		- 채널 크기 3: HSV (색조, 명도, 채도)
		
	※ 색 표현에 대한 참고
	- **색상 모델** (Color Model)
		- 색상 모델은 색을 수학적으로 표현하는 **추상적인 방법**입니다.
		- **예시:** RGB, CMYK, HSV 등.
		
	- **색 공간** (Color Space)
		- 색 공간은 특정 **색상 모델**을 기반으로, 특정 장치(모니터, 프린터 등)에서 재현할 수 있는 **색상의 구체적인 범위**를 정의한 것입니다. 
		- **예시:** sRGB, Adobe RGB (모두 RGB 색상 모델을 기반으로 함).
		
	- **채널** (Channel)
		- 채널은 색상 모델을 구성하는 **개별적인 요소**입니다.
		- **예시:** RGB의 R, G, B 채널
		
	- **채널공간** (Channel Space)
		- 이미지 처리 분야에서는 '채널 공간'이라는 용어가 **"작업 중인 색상 모델(예: RGB, HSV)과 그 채널들(예: R, G, B)"** 을 묶어서 표현하는 방식으로 사용되곤 합니다
		- 이는 다소 포괄적인 표현으로, 문맥에 따라 정확한 의미가 달라질 수 있습니다.
- ![[Pasted image 20250926131400.png]]
- 이 경우 **높이 4, 넓이 4, RGB 채널을 포함한 것**으로 **Shape 는 (4, 4, 3)** 으로 표현한다

### Convolutional Layer

- FC Layer(Dense Layer) 와의 차이점
	- **전역 패턴**(입력된 이미지의 모든 픽셀에 거친 패턴)을 학습
- Convolutionale Layer(합성곱 층)
	- **지역 패턴**을 학습
	
- **kernel**
	- Convolutionale Layer 는 Input Iamge 의 크기인 **특성 맵**(feature map)을 입력받게 된다.
	- **지역 패턴**을 학습하기 위해 이러한 특성 맵에 **커널**(kernel) 또는 **필터**(filter)라 불리는
	  **정사각형 행렬**을 적용하면 합성곱 연산을 수행하게 된다.
	- 커널의 경우 **3x3** , **5x5** 크기로 적용되는 것이 일반적
	- **스트라이드**(Stride)라 불리는 **지정된 간격**에 따라 순차적으로 이동한다
	- (출처: [youtube_신박AI채널](https://www.youtube.com/watch?v=lDqn1UNwgrY))![[Pasted image 20250926134207.png]]
	- (출처: [deeplearning.stanford.edu](http://deeplearning.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution/))![[Convolution_schematic.gif]] 
	- (출처: [youtube_신박AI채널](https://www.youtube.com/watch?v=lDqn1UNwgrY))![[Pasted image 20250926134815.png]]
	- kernel 의 크기가 다르면 출력되는 feature map의 크기도 달라진다
		- feature map 의 크기가 클수록 디테일을 표현할 수 있음
			- 그만큼 연산량이 증가하기 때문에 너무 크지 않도록 고려해야 됨
	- (출처: [youtube_신박AI채널](https://www.youtube.com/watch?v=lDqn1UNwgrY)) ![[Pasted image 20250926135752.png]]
	- stride 가 다를 경우, 같은 kernel 크기라도 출력되는 feature map 의 크기가 달라진다
		
	- kernel 의 크기와  stride 는 **연산량과 모델의 정확성 사이에서 밸런스**를 맞추어서 설정
		- kernel 과 stride 의 작용으로 나오는 feature map 의 크기가 원본보다 작아지는 
		  단점 존재
		- padding 사용으로 보완
	
- **Padding**
	- 원본의 크기가 작아지는 단점을 보환하기 위해 사용
	- 원본 이미지에 '**0**' 이라는 **padding 값을 채워** 이미지를 **확장한 후 합성곱 연산**을 적용
	- 출처([commons.wikimedia.org](https://modulabs.co.kr/blog/padding-cnn-basic))
	  ![[images_jaehyeong_post_d135f128-8e9d-43a2-bc18-e5221ebe82a3_cnn_padding.gif]]
	- 출처([wikipedia.org](https://en.wikipedia.org/wiki/Kernel_(image_processing)))
	  ![[2D_Convolution_Animation.gif]]
	 
- Activation Function(활성화 함수) - **ReLU**(Rectified Linear Unit)
	- **비선형성**을 도입
	- 모델이 복잡한 패턴을 학습할 수 있게 한다.
	- RuLU 의 경우 0 미만의 값은 0으로 출력하고 0 이상의 값은 그대로 출력함으로 
	  **gradient vanishing 문제**(기울기 소실)에 **덜 민감**함
		- 이 때문에 깊은 레이어에서도 효율적인 **역전파**(Back Propagation)가 가능하다
			
		- 일반적으로 **Sigmoid 함수**의 경우 **값을 0~1 사이로 정규화**시키는데 레이어가 깊어질 수록 **0.xxxx 의 값이 계속 미분**되게 되면 값이 점차 **0에 수렴**하게 되어 결국 **weight 값이 희미**해지는 **gradient vanishing 문제가 발생**하게 되다
	- 합성곱 연산을 통해 출력된 fearture map 의 경우 일반적으로 ReLU 활성화 함수를 거치게 되며, ReLU 함수가 **양의 값만을 활성화**하며 특징을 두드러지게 표현해준다
	- 활성화 함수는 여러개의 kernel 로 구성된 커널층에서 생성된 featura map 들의 점수를 확인해서 정리한다
		- 생성된 여러 feature map 들의 각 **요소들을 하나씩 활성화 함수를 통과**시켜, 
		  해당 활성화 **함수의 기준에 따라** 요소의 **값을 조정 $\cdot$ 억제 $\cdot$ 강조** 한다
		- 모든 feature map 의 요소 정리가 끝나면 그제서야 다름 레이어층의 kernel 층으로 다시 전달된다
			- 이 경우 **ReLU 함수**는 연관성이 없는 점수들(**양수가 아닌 값**)들을 
			  **모두 0으로 정리**해주는 필터링을 적용시킨다
	- (출처: [youtube_신박AI채널](https://www.youtube.com/watch?v=lDqn1UNwgrY)) ![[Pasted image 20250926143635.png]]
	- (출처:[wikipedia.org](https://en.wikipedia.org/wiki/Sigmoid_function))![[sigmoid_function.png]]
		
	- **활성화 함수의 그래프**는 특정 범위의 값들이 입력됐을 때, 각 **입력된 값들이 함수를 통과**했을 경우의 출력이 **어디쯤 위치하게 되는 지를 시각화**한 것!!
## Pooling Layer

- Convolutional Layer 와 유사하게 feature 