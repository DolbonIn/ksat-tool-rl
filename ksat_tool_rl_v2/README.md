# 수능 풀이 Tool-Calling RL v2
기준일: 2026-11-18. 제안 설계이며 실제 모델 학습 결과가 아니다.

## 결정
대량 8만 개 장문 SFT 선제작을 필수 단계에서 제외한다.
검증된 문제·정답 + 격리된 실행 도구 + 다중 턴 온라인 RL을 중심으로 만든다.
기존 모델의 도구 호출이 작동하면 RL-only를 먼저 시험한다. 호출 규약을 못 배우거나 정답 rollout이 거의 없을 때만 200~500개 수준의 실행 예시를 만드는 작은 warm-start를 시험한다. 이 수치는 시작 가정이지 검증된 최적값이 아니다.
RL은 해설 라벨 비용을 줄일 수 있으나 원문·정답·문제 다양성·계산 비용의 필요성을 없애지 않는다.

## 근거와 적용 범위
- ICRL (2026-03-09): rollout 프롬프트에 도구 사용 예시를 넣고 점차 제거하는 RL-only 방식. 검색 QA 중심 실험이며 수능 국어/영어에 대한 검증은 아님.
  https://arxiv.org/html/2603.08068v1
- Teaching Thinking Models to Reason with Tools (2026-05-07): 도구 추론 SFT+RL, no-tool 능력 유지와 안정적 RL의 중요성. 이 설계가 사용자의 소량 데이터에서 같은 성능을 보장하지 않음.
  https://arxiv.org/html/2605.06326v1
- ARCHER (2026-11-13): 소형 모델에서 실행 가능한 계산기 action trace와 보상 기반 개선. 수능 언어영역으로 자동 일반화되지 않으며 gold trace에 의존하는 부분은 그대로 채택하지 않음.
  https://arxiv.org/html/2611.18494v1
- TRL: 다중 턴 tools / environment_factory / GRPO.
  https://huggingface.co/docs/trl/grpo_trainer
- veRL: SGLang multi-turn rollout 및 assistant 토큰 마스킹.
  https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html
- NeMo Gym: RL 환경과 작은 SFT warm start.
  https://docs.nvidia.com/nemo/gym/v0.6.0/about/concepts/training/

## 1. 학습 환경
public problem → policy → 실제 tool executor → tool observation → policy → final → private grader → reward → GRPO update.

Policy는 공개 입력과 허용 도구만 본다. 정답·정오표·해설·gold evidence·보상 함수의 내부 상태는 grader 전용이다.
gold가 public JSON에 없다는 것만으로 안전하지 않다. 실행 도구 프로세스에는 정답 파일, 평가 데이터, 토큰·키·네트워크·학습기 메모리를 제공하지 않는다.
코드 실행을 학습기 안에서 eval/exec로 수행하지 않는다. OS/컨테이너 격리, 비특권 계정, 네트워크 차단, 읽기 전용 허용 파일, 실행 시간·메모리·프로세스·출력 제한을 적용한다.
episode 중 정답 여부를 알려 주는 check_answer 도구를 제공하지 않는다. 정답 판정은 final 제출 후 수행하고 episode를 종료한다.
문항별 환경 상태는 각 rollout 안에서만 유지되고 다음 rollout 또는 다른 후보와 공유하지 않는다.

## 2. 작은 도구 집합
math_v1:
- math_eval: 제한된 수식 DSL의 정확 산술/기호 계산.
- math_solve: 방정식·조건·정의역을 명시한 해 계산.
- 선택적 python_exec: 조합 열거·수치 확인 등 필요할 때만 별도 격리 런타임에서 제공.
  수치 확인과 샘플링은 증명이나 전체 경우 검사를 대신하지 않는다.

language_v1:
- find_spans: 현재 문항에 제공된 지문에서 검색.
- read_span: 지문의 특정 위치를 정확히 다시 읽기.
- 외부 웹 검색, 기출 정답 검색, 인터넷 사전 접근은 기본 환경에서 제외.
  짧은 지문이 이미 프롬프트에 있으면 도구를 쓰지 않고 풀어도 정상이다.
  검색 결과가 실제 존재함과 그 근거가 정답을 지지함은 서로 다른 검사다.

vision_v2:
- 현재 문제의 원본 그림에서 crop_region. 이미지 응답을 처리할 수 있는 VLM/processor 경로에만 사용.
- 듣기/ASR과 image reasoning은 별도의 입력·평가 트랙으로 시작한다.

배포 조건: 도구 사용 정확도는 '모델+도구 시스템'의 성능이다. 완전 온디바이스라면 필요한 런타임과 자산도 로컬에 있어야 한다. 학습용 Python 도구를 모바일에 배포할 수 없다면, 처음부터 모바일에서 구현 가능한 제한 DSL로 학습한다. teacher LLM은 배포 도구로 두지 않는다.

## 3. 데이터와 분할
초기 데이터 레코드: 고유 ID, family_id, 문제/지문/선지/이미지·음원, 공식 정답, 출처, 권리/정답/파싱 검수 상태, split, tools_profile.
장문 해설은 RL 시작의 필수 필드가 아니다.
정답을 모르는 미검수 문제는 완성된 RL 학습 데이터로 취급하지 않는다.

이전 연도 기준 분할안을 유지:
- 실제 시행 2021~2023: train
- 2024: validation
- 2025: test
같은 지문·짝수/홀수형·번역·숫자 변형·이미지 버전은 같은 family에 묶는다.
이전 자료의 3,087 train 자리는 예상 자리 수이고 확보된 RL 문제 수가 아니다.
그중 수학 966자리는 회차당 46자리 × 21회라는 계획 계산이며 원문/도형 확보 이후 실제 사용량을 센다.
기존 28개 해설은 2025 시행 test 쪽이며 원문/검수 부족 상태를 유지한다.

파일럿 제안:
- 원문·정답이 검수된 math train 500~1,000문제를 우선 확보.
- 문제당 rollout 8개.
- 언어영역은 별도 train 300~500문제로 시작해 근거 품질과 도구 이득 확인.
- warm-start SFT 0개가 첫 실험. 필요할 때만 200~500개.
- 원본 문제와 검증된 생성 문제를 섞을 때 생성기 가족도 train/test에서 분리.
이 수치는 모두 제안값이며 이미 제작된 수량이나 최소 필요량이 아니다.

## 4. RL-only → 필요 시 작은 warm-start
A. 원본 모델을 같은 문제/예산에서 no-tool 및 with-tool로 평가한다.
B. 도구 호출 예시 2개→1개→0개를 rollout prompt에 제공하는 커리큘럼을 시험한다.
C. 예시 감소는 고정 step보다 호출 유효성·정답률·완료율을 보고 진행한다.
D. 예시 제거 후에도 zero-shot 도구 호출이 유지되는지 별도 확인한다.
E. 호출을 거의 못 하거나 정답 rollout이 너무 희소하면, 작은 실행 예시 SFT 뒤 동일 RL과 비교한다.
도구 호출 예시를 넣는 것은 새로운 시험 문제의 정답을 알려 주는 것과 다르다. 예시·예시 지문도 train에 속해야 한다.

## 5. GRPO 설정과 올바른 마스킹
기본 실험은 동기식 online GRPO. 안정화 후 비동기 rollout을 검토한다.
그룹 내 8개 후보는 동일 problem_id, 공개 지문, asset, tool budget, 초기 환경 상태에서 시작한다. policy sampling seed만 다를 수 있다.
reset에서 문제를 후보마다 새로 뽑으면 서로 다른 난도의 문제를 같은 그룹으로 비교하게 된다.
tool outputs는 실제 runtime이 만들고 policy loss 대상에서 제외한다. assistant의 호출 이름·인자·설명·최종 응답은 policy action으로 학습한다.
생성 토큰 ID, 이전 policy logprob, role/action mask, policy version, 도구 실행 기록을 보존한다.
다중 턴을 재직렬화하면서 예전 reasoning을 버리거나 태그를 바꾸지 않게 tokenizer round-trip을 시험한다.
모든 후보가 동일한 정답 보상을 받는 그룹의 비율, 유효 호출률, 파싱 실패, truncation, 실제 처리 비용을 기록한다.
도구가 잘못 호출되었을 때 오류 관측을 보고 회복하는 것은 허용한다. 환경 장애로 생긴 실패와 모델의 잘못된 호출을 분리한다.
통신/인프라 실패는 답이 틀린 것으로 보상하지 않고 영향받은 그룹을 다시 수집하는 정책을 둔다.

초기 가정:
- num_generations: 8
- 공개 문제: 16개/optimizer-step 목표 → 128개 trajectory (메모리에 맞춰 microbatch 조정)
- assistant 생성 예산: 전체 episode 4096 tokens
- tool observation: 1회 최대 1024 tokens, episode 전체 4096 tokens
- tool call: 총 6회, assistant 턴: 8회
- 모델 context: 16384 tokens의 시작 가정; 실제 지원값과 입력 길이를 검증
- optimizer lr: LoRA 예시 1e-6~5e-6를 소규모 탐색; full FT와 동일 값으로 가정하지 않음
- teacher/judge: policy가 호출하지 못하는 offline/학습 전용
- 예산 소진: final 없음/도중 종료를 완료로 처리하지 않음
위 키는 프레임워크에 바로 넣는 설정이 아니라 구현해야 할 환경 정책이다.

## 6. 보상
Phase 1:
- 정상 종료 + 공식 정답 일치: 1
- 정상 종료 + 오답: 0
- final 누락, 최종 포맷 불능, 금지 동작: -1
- 호출 횟수나 장문 출력 자체에는 양의 보상을 주지 않는다.
- 실행 인프라 장애는 보상 0과 구별하고 재수집한다.

Phase 2 제안:
R = A + 0.1*A*E - C
A: 공식 정답 일치(0/1).
E: 별도 검수로 신뢰성이 확인된 근거/관측 일관성 점수(0~1). 아직 신뢰할 수 없으면 0.
C: 0~0.05로 제한한 정규화 실행 비용. 실제 runtime 로그에서 계산한다.
틀린 답이 겉보기 형식/도구 호출 보너스로 맞는 답보다 높은 보상을 얻지 못하게 한다.
초기에는 비용 패널티를 0으로 두고 성공 사례를 확보한 후 효율화를 시험한다.
단순히 존재하는 문장을 인용했다는 사실, '검산했다'는 문구, tool call의 존재만으로 E를 주지 않는다.
수학 정답 보상이 해설의 엄밀성을 보장하지는 않는다. 언어영역은 공식 정답 정확도와 근거 품질의 지표/표본 검수를 분리한다.
상세 교육용 해설은 승인된 실행 기록과 최종 답을 바탕으로 별도 작성·검수하며, 이때 정답을 본 provenance를 남긴다.

## 7. 최소 실험군
동일 held-out 문제, 동일 총 예산과 배포조건에서:
A. 원본 모델 no-tool
B. 원본 모델 + tools (학습 없음)
C. RL-only + tools
D. 작은 warm-start + RL + tools
E. RL 모델 no-tool 회귀 검사
문항/지문군 기준 bootstrap 등으로 차이의 불확실성도 보고한다.
pass@1, 과목·유형별 정확도, 도구 호출 성공률, 잘못된 근거율, 평균/p95 token·지연, truncation, 양자화 후 정확도를 함께 기록한다.
도구 호출률 상승은 성공 지표가 아니다.
반복 rollout은 새 문제가 아니다. 예: 1,000문제×8후보×평균 2,000 생성 tokens = 1,600만 생성 tokens/전체 문제 1회 순회. tool observation/prefill/optimization 비용은 별도다.

## 8. 프레임워크
파일럿: TRL GRPOTrainer의 tools 또는 environment_factory.
수능처럼 corpus가 정해진 경우 외부 dataset 행/공유 키로 동일 group 초기 상태를 고정한다.
reset/get_reward 외 공개 메서드 노출과 reward 합산 정책을 설치 버전에서 확인한다.
최신 문서도 실험적 API 변경 가능성을 명시하므로 버전·commit·모델 parser를 고정한다.
SGLang 기반 다중 GPU rollout이 중요하면 veRL의 multi-turn 경로를 다음 단계로 사용한다.
이 문서와 YAML은 실제 training launch를 수행한 코드가 아니다.

## 패키지 구현 범위
- 이 설계서
- 프레임워크 중립 recipe
- 자작 episode 레코드 예시
- bounded reward 함수와 9개 단위 검사 결과
실제 tool runtime, sandbox, rollout adapter, GRPO 학습은 구현/실행하지 않았다.
