"""Reference bounded reward only; not a semantic checker or RL trainer.
`None` means infrastructure failure: resample/exclude the affected group, not reward 0.
All fields must come from trusted runtime/grader, never the model's self-report.
"""
from dataclasses import dataclass
from typing import Optional
import math

@dataclass(frozen=True)
class Outcome:
    correct: bool
    complete: bool = True
    final_protocol_valid: bool = True
    forbidden_action: bool = False
    infrastructure_error: bool = False
    verified_evidence_score: Optional[float] = None
    normalized_runtime_cost: float = 0.0

def reward(o: Outcome, efficiency_phase: bool = False) -> Optional[float]:
    if o.infrastructure_error:
        return None
    if not o.complete or not o.final_protocol_valid or o.forbidden_action:
        return -1.0
    for name, value in [
        ("verified_evidence_score", o.verified_evidence_score),
        ("normalized_runtime_cost", o.normalized_runtime_cost),
    ]:
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"{name} must be finite and within [0,1]")
    a = float(o.correct)
    if not efficiency_phase:
        return a
    e = o.verified_evidence_score if o.verified_evidence_score is not None else 0.0
    return a + 0.1 * a * e - 0.05 * o.normalized_runtime_cost
