"""Shared scoring logic for per-problem status and aggregate computation.

Single source of truth used by ProgressReporter (validator) and
test_runner (local testing). Do NOT reimplement scoring elsewhere.
"""

from typing import Optional

from src.agent.types import AggregateScore, ScoreDict


def is_problem_successful(score_dict: Optional[ScoreDict], category: str) -> bool:
    """Determine if a problem was successfully solved.

    Uses category-aware criteria:
    - product: rule >= 1
    - shop: rule >= 1 AND shop >= 1
    - voucher: rule >= 1 AND budget >= 1

    Args:
        score_dict: Score dictionary from ProblemScorer.score_problem().
            None means scoring failed entirely.
        category: Problem category ("product", "shop", "voucher").

    Returns:
        True if the agent successfully solved the problem.
    """
    if score_dict is None:
        return False

    rule_ok = score_dict.get("rule", 0) >= 1
    category = category.lower()

    if category == "shop":
        return rule_ok and score_dict.get("shop", 0) >= 1
    elif category == "voucher":
        return rule_ok and score_dict.get("budget", 0) >= 1
    else:
        return rule_ok


def compute_aggregate(
    results: list[dict],
    total_problems: int,
) -> AggregateScore:
    """Compute aggregate score from per-problem results.

    Denominator is always total_problems (full suite size).
    Unscored problems (timeouts, crashes) count as failures.

    Args:
        results: List of dicts, each with "category" and "score_dict" keys.
            score_dict may be None (scoring failed).
        total_problems: Total problems in the suite (denominator).

    Returns:
        Dict with ground_truth_rate, success_rate, format_score,
        field_matching, total_problems, successful_problems, scored_problems.
    """
    total = max(total_problems, 1)

    gt_successes = success_count = scored = 0
    format_total = rule_total = 0.0

    for r in results:
        sd = r["score_dict"]
        if sd is None:
            continue
        scored += 1
        if sd.get("gt", 0) >= 1:
            gt_successes += 1
        if is_problem_successful(sd, r["category"]):
            success_count += 1
        format_total += sd.get("format", 0)
        rule_total += sd.get("rule", 0)

    return {
        "ground_truth_rate": min(gt_successes / total, 1.0),
        "success_rate": min(success_count / total, 1.0),
        "format_score": min(format_total / total, 1.0),
        "field_matching": min(rule_total / total, 1.0),
        "total_problems": total_problems,
        "successful_problems": success_count,
        "scored_problems": scored,
    }


# Reasoning coefficient model: final_score = success_rate * coefficient
# coefficient ranges from COEFF_FLOOR (zero reasoning) to 1.0 (good reasoning)
# Agents scoring above COEFF_CEILING_THRESHOLD get full credit (coefficient 1.0)
COEFF_FLOOR = 0.3
COEFF_CEILING_THRESHOLD = 0.80


def reasoning_coefficient(reasoning_quality: float) -> float:
    """Compute the reasoning coefficient from reasoning quality.

    Maps reasoning_quality to a coefficient:
    - 0.0 -> COEFF_FLOOR (0.3)
    - COEFF_CEILING_THRESHOLD (0.80) -> 1.0
    - Anything above 0.80 -> 1.0 (full credit)

    Args:
        reasoning_quality: Average reasoning quality (0.0 - 1.0).

    Returns:
        Coefficient between COEFF_FLOOR and 1.0.
    """
    quality = max(0.0, min(1.0, reasoning_quality))
    if quality >= COEFF_CEILING_THRESHOLD:
        return 1.0
    return round(COEFF_FLOOR + quality * (1.0 - COEFF_FLOOR) / COEFF_CEILING_THRESHOLD, 4)


def blend_final_score(success_rate: float, reasoning_quality: float) -> float:
    """Compute final score as success_rate * reasoning_coefficient.

    A regex agent with zero reasoning gets at most success_rate * 0.3.
    An agent with perfect reasoning gets full credit for its outcome score.

    Args:
        success_rate: Outcome-based success rate (0.0 - 1.0).
        reasoning_quality: Aggregate reasoning quality (0.0 - 1.0).

    Returns:
        Final score (0.0 - 1.0).
    """
    coeff = reasoning_coefficient(reasoning_quality)
    return round(success_rate * coeff, 4)
