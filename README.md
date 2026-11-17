# KSAT Tool-Calling RL

수능·고3 모의고사 국어·수학·영어 온디바이스 모델을 위한 **Tool-calling RL 중심 설계 및 데이터 검사 도구**.

## 우선순위

1. **기본 방향:** 검수된 문제·공식 정답과 실제 도구 실행 환경을 이용하는 다중 턴 RL.
2. **조건부 보완:** 도구 호출이 제대로 작동하지 않을 때만 소규모 warm-start SFT를 비교.
3. **선택적 확장:** 데이터셋 제작 명세의 풀이·검증·수정 SFT를 별도 실험으로 활용.

`ksat_reasoning_blueprint/README.md`의 8만 개 SFT 계획은 이전 확장안이며, RL 시작의 선행 조건이 아닙니다. 우선 적용할 의사결정은 `ksat_tool_rl_v2/README.md`입니다. 두 기존 패키지의 파일은 원래 경로와 내용 그대로 보존했습니다.

## 현재 구현 범위

| 구성 | 상태 |
|---|---|
| Tool-calling RL 설계, 프레임워크 중립 설정, episode 예제 | 포함 |
| 정답·근거·비용을 분리한 참고 보상 함수 | 포함; 실제 채점기 아님 |
| 데이터 스키마, 생성·검증·수정 프롬프트 | 포함 |
| 텍스트 데이터 검사 및 SFT/RL export | 포함 |
| 기존 해설 초안 | 28개; 원문 미확보·미승인 상태 유지 |
| 실제 기출 원문·도형·음원 | 포함하지 않음 |
| 실제 도구 sandbox, 학습기 연동, GRPO 학습 실행 | 미구현/미실행 |
| 검수 완료 학습용 기출 | 0개 |

이 저장소는 완성된 학습 파이프라인이나 완성 기출 데이터셋이 아닙니다. 논문·모델·버전 관련 기존 문서의 주장과 출처는 원문 보존이며, 이번 저장소 패키징에서 재검증하지 않았습니다.

## 구조

```text
ksat_tool_rl_v2/           # 기본 RL 설계·보상 함수·설정·episode 예제
ksat_reasoning_blueprint/ # 선택적 SFT 확장안·스키마·프롬프트·검사/exporter
scripts/run_checks.sh     # 두 구성의 단위 테스트
scripts/publish.sh        # 사용자의 gh 인증으로 비공개 신규 저장소 생성/push
scripts/verify_manifest.py # 패키지 파일 SHA-256 대조
checks/test_reward.py     # 재실행 가능한 보상 함수 단위 테스트
reports/                  # 이번 패키징의 실제 로컬 검사 결과
PACKAGE_MANIFEST.json     # 파일별 무결성 목록; GitHub 업로드 기록이 아님
```

## 로컬 검사

Python 3.10 이상을 사용하세요.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
bash scripts/run_checks.sh
python3 scripts/verify_manifest.py
```

테스트는 작은 합성 fixture와 기존 28개 초안의 차단 상태를 검사합니다. 테스트 통과는 기출 정답·해설의 사람 검수나 실제 학습 성능을 의미하지 않습니다.

## GitHub 업로드

**이 패키지를 만드는 과정에서는 원격 저장소를 만들거나 push하지 않았습니다.** 현재 채팅의 GitHub 연결에 쓰기 도구가 없어, 사용자가 인증한 GitHub CLI에서 실행할 스크립트를 제공합니다.

Git과 GitHub CLI(`gh`)가 설치되고 `DolbonIn` 계정으로 인증된 환경에서 실행하세요. 인증 토큰을 대화나 저장소에 붙여 넣지 마세요.

```bash
gh auth login
bash scripts/publish.sh
```

기본 대상은 **`DolbonIn/ksat-tool-rl` 비공개 신규 저장소**입니다. 다른 계정, 동일 이름의 기존 저장소, 이미 초기화된 로컬 Git 저장소가 발견되면 중단합니다. 공개 저장소 생성, 기존 remote 변경, force-push, 기존 저장소 덮어쓰기는 하지 않습니다.

선택적으로 저장소 이름만 바꿀 수 있습니다.

```bash
bash scripts/publish.sh ksat-tool-rl-experiments
```

## 데이터 경계

공식 정답은 grader 전용으로 분리하며 모델 또는 도구 sandbox에 노출하지 않습니다. 평가 문제와 변형·동일 지문은 학습에서 분리합니다. 원문·해설의 이용권과 품질 검수는 별도의 확인 사항입니다. 이 패키지는 기존 자료의 권리를 새로 부여하거나 오픈소스 라이선스를 지정하지 않습니다.
