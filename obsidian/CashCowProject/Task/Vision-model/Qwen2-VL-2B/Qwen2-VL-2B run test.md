테스트 환경

세팅 전
![[Pasted image 20251117201144.png]]

hugging face - requirement 권장 명령어 실행

![[Pasted image 20251117202622.png]]
![[Pasted image 20251117202846.png]]

세팅 후
![[Pasted image 20251117202927.png]]


## Test code 작성 및 실행
``` python
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch

model_id = "Qwen/Qwen2-VL-2B-Instruct"

processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

img = Image.open("test_data/image/article_test1.jpg")

prompt = "이 이미지의 기사를 그대로 읽고 요약해줘."

inputs = processor(text=prompt, images=img, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=200)

result = processor.decode(output[0], skip_special_tokens=True)
print(result)
```

![[Pasted image 20251117220022.png]]
torch 의 부재로 import error 발생

torch 설치 후 재실행
![[Pasted image 20251117220043.png]]
![[Pasted image 20251117222219.png]]

![[Pasted image 20251117222259.png]]
Chat GPT 문의 결과

**Transformers 최신(dev) 버전과 Qwen2-VL Processor 로딩 시 생기는 구조적 문제**
(즉, HF 쪽 코드 문제)

---

### 오류 원인 요약 (중요)

#### Transformers dev 버전은 새 video_processor automatic-loading 로직이 들어갔는데

#### 설치한 transformers 버전은 “video extractor 목록(extractors)”이 비어 있음(None)

#### 그래서 AutoProcessor가 video processor를 자동 추론하다가 NoneType을 순회하면서 터진 것.

즉,

> **Qwen2-VL 모델은 video processor 구성 요소가 없는데  
> Transformers dev는 ‘video_processor 자동 로드’를 시도하면서 충돌함.**

HF master 브랜치에서 이미 보고된 문제.

#### 솔루션 3가지 중 택1

1. transformers 버전을 “조금 더 안정적인 dev 릴리즈”로 다운그레이드
2. dev 버전을 유지하되, “video 처리 자동감지” 기능 비활성화
3. qwen-vl-utils 버전 올리기 (문제 완화)

3 번 솔루션은 버전 밸런싱 과정에서 오류가 더 발생할 가능성도 존재하므로 1 과 2 중 선택하는 것을 권장

#### 1 번 솔루션 적용
![[Pasted image 20251117222949.png]]
![[Pasted image 20251117223020.png]]
![[Pasted image 20251117223032.png]]

![[Pasted image 20251117223048.png]]

![[Pasted image 20251117223111.png]]

충돌 문제는 해결되었고, accelerate 부재 오류 확인

accelerate 설치 후 실행
![[Pasted image 20251117223211.png]]
![[Pasted image 20251117223311.png]]

![[Pasted image 20251117223326.png]]
모델을 불러오기까지 성공

![[Pasted image 20251117224542.png]]


