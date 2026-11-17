# 수정
입력: 원문, 이전 시도, 구체적인 검증기 피드백.
오류를 수정하고 독립적으로 읽을 수 있는 새 풀이를 작성하라. 답에 맞춰 근거를 꾸며내지 마라.
미해결이면 uncertain으로 남겨라.
출력: {status,revised_answer,revised_rationale,corrected_step_ids,evidence,verification}
수정본도 초기 후보와 같은 검증 절차를 다시 통과해야 한다.
정답을 본 수정본은 gold_visible=true로 기록하고 repair task로 보존한다.
이를 정답을 보지 않은 blind-solve 예제로 무표시 변경하지 않는다.
