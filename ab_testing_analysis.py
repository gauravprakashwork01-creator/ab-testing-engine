"""A/B testing analysis tool.

Simulates control and treatment cohorts, computes conversion metrics,
uplift, confidence intervals, and Welch's t-test significance.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, pi, sqrt
from random import Random
from typing import List, Sequence


@dataclass(frozen=True)
class ExperimentResult:
    control_rate: float
    treatment_rate: float
    uplift: float
    ci_low: float
    ci_high: float
    t_stat: float
    p_value: float
    decision: str


def simulate_group(sample_size: int, conversion_probability: float, rng: Random) -> List[int]:
    """Generate a binary conversion list for one cohort."""
    if sample_size <= 0:
        raise ValueError("sample_size must be > 0")
    if not 0 <= conversion_probability <= 1:
        raise ValueError("conversion_probability must be between 0 and 1")

    return [1 if rng.random() < conversion_probability else 0 for _ in range(sample_size)]


def simulate_experiment(
    control_size: int,
    treatment_size: int,
    control_probability: float,
    treatment_probability: float,
    seed: int = 7,
) -> tuple[List[int], List[int]]:
    """Simulate control and treatment conversion outcomes."""
    rng = Random(seed)
    control = simulate_group(control_size, control_probability, rng)
    treatment = simulate_group(treatment_size, treatment_probability, rng)
    return control, treatment


def conversion_rate(group: Sequence[int]) -> float:
    """Return mean conversion for a binary cohort."""
    if not group:
        raise ValueError("group cannot be empty")
    return sum(group) / len(group)


def uplift(control_rate: float, treatment_rate: float) -> float:
    """Absolute uplift (treatment - control)."""
    return treatment_rate - control_rate


def _variance(group: Sequence[int]) -> float:
    mean = conversion_rate(group)
    n = len(group)
    if n < 2:
        raise ValueError("group must have at least 2 observations")
    return sum((x - mean) ** 2 for x in group) / (n - 1)


def welch_t_test(control: Sequence[int], treatment: Sequence[int]) -> tuple[float, float]:
    """Compute Welch's t-statistic and a two-sided p-value (normal approx)."""
    control_mean = conversion_rate(control)
    treatment_mean = conversion_rate(treatment)

    control_var = _variance(control)
    treatment_var = _variance(treatment)

    se = sqrt((control_var / len(control)) + (treatment_var / len(treatment)))
    if se == 0:
        return 0.0, 1.0

    t_stat = (treatment_mean - control_mean) / se
    # Large-sample normal approximation for two-sided p-value.
    p_value = 2 * (1 - _normal_cdf(abs(t_stat)))
    return t_stat, p_value


def _normal_cdf(x: float) -> float:
    return (1 + erf(x / sqrt(2))) / 2


def confidence_interval_uplift(
    control: Sequence[int], treatment: Sequence[int], confidence: float = 0.95
) -> tuple[float, float]:
    """Wald confidence interval for absolute uplift."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    control_rate = conversion_rate(control)
    treatment_rate = conversion_rate(treatment)

    se = sqrt(
        (control_rate * (1 - control_rate) / len(control))
        + (treatment_rate * (1 - treatment_rate) / len(treatment))
    )

    z = 1.96 if confidence == 0.95 else _z_for_confidence(confidence)
    delta = z * se
    diff = treatment_rate - control_rate
    return diff - delta, diff + delta


def _z_for_confidence(confidence: float) -> float:
    # Rational approximation using inverse error function via Newton iterations.
    target = (1 + confidence) / 2
    x = 0.0
    for _ in range(30):
        fx = _normal_cdf(x) - target
        dfx = (1 / sqrt(2 * pi)) * exp(-(x**2) / 2)
        x -= fx / dfx
    return x


def interpret_decision(p_value: float, uplift_value: float, alpha: float = 0.05) -> str:
    """Recommend ship decision based on significance + positive uplift."""
    if p_value < alpha and uplift_value > 0:
        return "Ship"
    return "Do not ship"


def analyze_experiment(control: Sequence[int], treatment: Sequence[int]) -> ExperimentResult:
    """Run the full experiment analysis pipeline."""
    control_rate = conversion_rate(control)
    treatment_rate = conversion_rate(treatment)
    uplift_value = uplift(control_rate, treatment_rate)
    ci_low, ci_high = confidence_interval_uplift(control, treatment)
    t_stat, p_value = welch_t_test(control, treatment)
    decision = interpret_decision(p_value, uplift_value)

    return ExperimentResult(
        control_rate=control_rate,
        treatment_rate=treatment_rate,
        uplift=uplift_value,
        ci_low=ci_low,
        ci_high=ci_high,
        t_stat=t_stat,
        p_value=p_value,
        decision=decision,
    )


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    control, treatment = simulate_experiment(
        control_size=10_000,
        treatment_size=10_000,
        control_probability=0.12,
        treatment_probability=0.135,
        seed=42,
    )
    result = analyze_experiment(control, treatment)

    print("A/B TEST RESULTS")
    print("-" * 40)
    print(f"Control conversion rate:   {format_percent(result.control_rate)}")
    print(f"Treatment conversion rate: {format_percent(result.treatment_rate)}")
    print(f"Absolute uplift:           {format_percent(result.uplift)}")
    print(
        "95% CI for uplift:         "
        f"[{format_percent(result.ci_low)}, {format_percent(result.ci_high)}]"
    )
    print(f"Welch t-statistic:         {result.t_stat:.4f}")
    print(f"p-value (two-sided):       {result.p_value:.6f}")
    print(f"Decision:                  {result.decision}")


if __name__ == "__main__":
    main()
