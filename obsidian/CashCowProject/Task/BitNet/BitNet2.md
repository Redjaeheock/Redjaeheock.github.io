### BitNet 성능 확인

git hub 에서 benchmark 파일을 통해 성능을 측정을 가능하게 마련해줬다.
```
usage: e2e_benchmark.py -m MODEL [-n N_TOKEN] [-p N_PROMPT] [-t THREADS]  
   
Setup the environment for running the inference  
   
required arguments:  
  -m MODEL, --model MODEL  
                        Path to the model file. 
   
optional arguments:  
  -h, --help  
                        Show this help message and exit. 
  -n N_TOKEN, --n-token N_TOKEN  
                        Number of generated tokens. 
  -p N_PROMPT, --n-prompt N_PROMPT  
                        Prompt to generate text from. 
  -t THREADS, --threads THREADS  
                        Number of threads to use. 
```
- `-m`, `--model`: 모델 파일의 경로입니다. 스크립트를 실행할 때 반드시 제공해야 하는 인수입니다.
- `-n`, `--n-token`: 추론 중 생성할 토큰 수입니다. 기본값은 128인 선택적 인수입니다.
- `-p`, `--n-prompt`: 텍스트 생성에 사용할 프롬프트 토큰 수입니다. 기본값은 512인 선택적 인수입니다.
- `-t`, `--threads`: 추론을 실행하는 데 사용할 스레드 수입니다. 기본값은 2인 선택적 인수입니다.
- `-h`, `--help`: 도움말 메시지를 표시하고 종료합니다. 이 인수를 사용하면 사용법 정보를 표시할 수 있습니다.

```bash
python utils/e2e_benchmark.py -m /path/to/model -n 200 -p 256 -t 4
```
이 명령은 에 있는 모델을 사용하여 추론 벤치마크를 실행하고 `/path/to/model`, 4개의 스레드를 활용하여 256개 토큰 프롬프트에서 200개 토큰을 생성한다.


테스트 - 입력 토큰 수 32, 출력 토큰 수 64, 스레드 크기 (4 | 8)

![[화면 캡처 2025-08-16 224518.png]]
해석
- **pp64 (입력 프롬프트 처리)**
    - 4스레드: 4.92 tok/s
    - 8스레드: 5.47 tok/s (**↑ 빨라짐**)  
        → 입력은 한 번에 여러 토큰을 **병렬 연산**하니까 스레드 늘린 효과가 뚜렷하게 나타남.
        
- **tg32 (출력 토큰 생성)**
    - 4스레드: 5.23 tok/s
    - 8스레드: 5.03 tok/s (**↓ 느려짐**)  
        → 출력은 **순차적(앞 토큰 → 다음 토큰)** 구조라 병렬화 여지가 제한적.  
        → 오히려 스레드가 많아지면서 **스레드 관리/캐시 충돌/동기화 오버헤드**가 커져 성능이 살짝 떨어진 것.

개인적 견해
- 로컬의 코어 : 스레드 구조가 1 : 2 구조라서 입력 처리는 속도가 증가한 반면, 출력은 감소한 것으로 유추
- 스레드 옵션은 하드웨어의 **코어 수도 참고**하면서 **스레드 수**를 **설정**하는 것이 좋을 것으로 판단됨

[ GPT-5 ]
#### 실사용 관점에서

- 대화형에서 **체감은 주로 tg 쪽**이에요 → 답변 생성 속도.
    
- 따라서 프롬프트 길이가 아주 길지 않다면 **스레드를 무작정 늘리기보다는 tg 속도가 좋은 값**(보통 4~6스레드 사이)을 찾는 게 이득입니다


\[ **참고 내용** ]
#### `pp64` (prompt processing 64)

- **뜻**: _Prompt Processing with 64 tokens_
- 모델이 입력 프롬프트(맥락 텍스트) **64 토큰을 한꺼번에 집어넣었을 때의 처리 속도**
- 특징:
    - 프롬프트는 한 번에 여러 토큰을 병렬로 처리할 수 있어요 (병렬 행렬곱).
    - 따라서 **병렬화 효과가 잘 드러남** 
	    → 스레드 늘리면 보통 속도가 올라가는 쪽.
    - 실제 대화에서 "질문을 던질 때" 모델이 입력 텍스트를 읽는 속도와 연관.

#### `tg32` (text generation 32)

- **뜻**: _Text Generation with 32 tokens_
- 모델이 **출력 토큰을 순차적으로 32개 생성**할 때의 속도
- 특징:
    - 생성은 오토리그레시브(auto-regressive)라서 **앞 토큰이 나와야 다음 토큰을 계산** 가능
    - 즉, **병렬화가 제한적**
	    → 스레드 늘려도 크게 빨라지지 않음.
    - 심지어 **동기화** - **캐시** **경쟁** 때문에 **오버헤드**가 커져서 속도가 줄기도 함.
    - 실제 대화에서 "답변을 이어서 생성할 때" 모델이 내보내는 토큰 속도와 직접적으로 연관.

---

### e2e_benchmark.py 통해 입출력 토큰 설정에 따른 성능 테스트

| model                                | size     | params | backend | threads | n_batch | pp  | tg  | pp(t/s)     | tg(t/s)     |
| ------------------------------------ | -------- | ------ | ------- | ------- | ------- | --- | --- | ----------- | ----------- |
| bitnet-b1.58 2B I2_S - 2 bpw ternary | 1.71 GiB | 2.74 B | CPU     | 4       | 1       | 16  | 32  | 5.43 ± 0.05 | 5.16 ± 0.40 |
| bitnet-b1.58 2B I2_S - 2 bpw ternary | 1.71 GiB | 2.74 B | CPU     | 4       | 1       | 16  | 64  | 5.36 ± 0.03 | 5.20 ± 0.29 |
| bitnet-b1.58 2B I2_S - 2 bpw ternary | 1.71 GiB | 2.74 B | CPU     | 4       | 1       | 16  | 128 | 5.24 ± 0.03 | 5.18 ± 0.12 |
| bitnet-b1.58 2B I2_S - 2 bpw ternary | 1.71 GiB | 2.74 B | CPU     | 4       | 1       | 16  | 256 | 5.43 ± 0.16 | 5.17 ± 0.08 |
| bitnet-b1.58 2B I2_S - 2 bpw ternary | 1.71 GiB | 2.74 B | CPU     | 4       | 1       | 32  | 32  | 5.34 ± 0.45 | 5.47 ± 0.10 |
| bitnet-b1.58 2B I2_S - 2 bpw ternary | 1.71 GiB | 2.74 B | CPU     | 4       | 1       | 32  | 64  | 5.20 ± 0.37 | 5.25 ± 0.27 |
| bitnet-b1.58 2B I2_S - 2 bpw ternary | 1.71 GiB | 2.74 B | CPU     | 4       | 1       | 32  | 128 | 5.22 ± 0.33 | 5.20 ± 0.15 |
| bitnet-b1.58 2B I2_S - 2 bpw ternary | 1.71 GiB | 2.74 B | CPU     | 4       | 1       | 32  | 256 | 5.27 ± 0.52 | 5.11 ± 0.13 |
| bitnet-b1.58 2B I2_S - 2 bpw ternary | 1.71 GiB | 2.74 B | CPU     | 4       | 1       | 64  | 32  | 5.32 ± 0.46 | 5.35 ± 0.54 |
| bitnet-b1.58 2B I2_S - 2 bpw ternary | 1.71 GiB | 2.74 B | CPU     | 4       | 1       | 64  | 64  | 5.28 ± 0.19 | 5.22 ± 0.22 |
| bitnet-b1.58 2B I2_S - 2 bpw ternary | 1.71 GiB | 2.74 B | CPU     | 4       | 1       | 64  | 128 | 5.36 ± 0.21 | 5.15 ± 0.11 |
| bitnet-b1.58 2B I2_S - 2 bpw ternary | 1.71 GiB | 2.74 B | CPU     | 4       | 1       | 64  | 256 | 5.27 ± 0.38 | 4.28 ± 2.00 |
| bitnet-b1.58 2B I2_S - 2 bpw ternary | 1.71 GiB | 2.74 B | CPU     | 4       | 1       | 128 | 32  | 5.12 ± 0.14 | 5.12 ± 0.42 |
| bitnet-b1.58 2B I2_S - 2 bpw ternary | 1.71 GiB | 2.74 B | CPU     | 4       | 1       | 128 | 64  | 5.13 ± 0.16 | 5.24 ± 0.22 |
| bitnet-b1.58 2B I2_S - 2 bpw ternary | 1.71 GiB | 2.74 B | CPU     | 4       | 1       | 128 | 128 | 4.78 ± 0.15 | 4.89 ± 0.19 |
| bitnet-b1.58 2B I2_S - 2 bpw ternary | 1.71 GiB | 2.74 B | CPU     | 4       | 1       | 128 | 256 | 5.14 ± 0.09 | 5.00 ± 0.06 |
| bitnet-b1.58 2B I2_S - 2 bpw ternary | 1.71 GiB | 2.74 B | CPU     | 4       | 1       | 256 | 32  | 5.17 ± 0.09 | 3.96 ± 2.21 |
| bitnet-b1.58 2B I2_S - 2 bpw ternary | 1.71 GiB | 2.74 B | CPU     | 4       | 1       | 256 | 64  | 5.17 ± 0.09 | 5.26 ± 0.25 |
| bitnet-b1.58 2B I2_S - 2 bpw ternary | 1.71 GiB | 2.74 B | CPU     | 4       | 1       | 256 | 128 | 5.18 ± 0.10 | 5.08 ± 0.13 |
| bitnet-b1.58 2B I2_S - 2 bpw ternary | 1.71 GiB | 2.74 B | CPU     | 4       | 1       | 256 | 256 | 4.72 ± 0.27 | 4.93 ± 0.28 |
요약: ???

---
위의 실행 이미지를 보면 

```bash
warning: asserts enabled, performance may be affected
warning: debug build, performance may be affected
```

두 가지 경고문이 발생하는 것을 확인할 수 있다.

`경고: asserts 활성화, 성능에 영향을 미칠 수 있습니다`
`경고: 디버그 빌드, 성능에 영향을 미칠 수 있습니다`

위 두 가지 내용은 cmake 에서 확인할 수 있는 내용이라고 하니, 한 번 살펴보자 한다.
	
( 어떤 파일인지 모르니 일단 아.묻.따 찾기 )

![[Pasted image 20250818201408.png]]
debug build 를 통해서 llama-bench.cpp 파일에 관련 출력문이 있는 것을 확인할 수 있었다(운이 좋았다)

그럼 조회된 문장을 통해서 파일의 위치를 확인한 후, 해당 위치를 열어보자

![[화면 캡처 2025-08-18 202751.png]]