# 독립 검증
입력: 원문, authoritative gold(있는 경우), candidate solution. 초기 solver와 독립된 호출이다.
정답 일치와 풀이 타당성을 따로 판단한다. 답이 맞아도 잘못된 근거면 탈락이다.
수학: 정의역, 동치변형, 경우 누락, 존재/유일성, 계산을 검사한다. 수치 대입 성공을 증명으로 취급하지 않는다.
국어/영어: 근거 span의 실제 존재, 양화/범위/인과/주체/시점 보존, 선지별 근거를 검사한다.
공식 정답과 불일치하면 공식 정답을 다수결로 변경하지 말고 정오표 확인 대상으로 분리한다.
출력:
{input_valid,answer_pass,rationale_pass,invalid_steps,unsupported_claims,evidence_errors,
 ambiguity,decision:accept|repair|quarantine,report}
단일 신뢰도 숫자만으로 승인하지 않는다. 오류 위치와 이유를 기록한다.
