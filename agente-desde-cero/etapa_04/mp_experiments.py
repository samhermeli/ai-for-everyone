"""
Experiment runner — el orquestador completo.

Corre las 3 variantes del agente contra:
  - Skill evals (de mp_evals.py)
  - Convergence eval (de mp_convergence.py)

Imprime un dashboard comparativo y persiste resultados como JSON en results/.

Concepto (L11): evaluation-driven development.
  Cada variante es una "version" del agente. El experiment runner aplica
  todos los evaluadores a todas las variantes y produce una tabla
  apples-to-apples — es lo que permite decidir empiricamente que cambio
  al prompt o modelo realmente mejoro la calidad del agente.

Standalone:
    export $(cat ../.env | xargs) && python mp_experiments.py
"""

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

from mp_agent import ALL_VARIANTS, AgentVariant
from mp_convergence import (
    ConvergenceResult,
    evaluate_convergence,
    print_convergence_report,
)
from mp_evals import run_skill_evals_for_variant


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# ---------------------------------------------------------------------------
# Resultado de un experimento (una variante corrida contra todos los evals)
# ---------------------------------------------------------------------------
@dataclass
class ExperimentResult:
    variant_name: str
    timestamp: str

    # Skill evals
    code_score: float
    router_score: float
    clarity_score: float

    # Convergence eval
    convergence_score: float
    optimal_path: int
    convergence_successful: int
    convergence_failed: int

    # Token consumption
    total_tokens: int
    avg_tokens_per_case: int


def run_experiment(variant: AgentVariant) -> tuple[ExperimentResult, dict, ConvergenceResult]:
    """
    Una corrida completa de una variante: skills + convergencia.
    Devuelve el ExperimentResult agregado + los resultados completos
    para inspeccion (los completos NO se mandan al dashboard pero si al JSON).
    """
    t0 = time.time()
    print(f"\n  ─── Skill evals para '{variant.name}' ───", flush=True)
    skill = run_skill_evals_for_variant(variant)
    print(f"  skill evals listos en {time.time() - t0:.0f}s", flush=True)

    t1 = time.time()
    print(f"\n  ─── Convergence eval para '{variant.name}' ───", flush=True)
    conv = evaluate_convergence(variant)
    print(f"  convergence eval listo en {time.time() - t1:.0f}s", flush=True)

    result = ExperimentResult(
        variant_name=variant.name,
        timestamp=datetime.now(timezone.utc).isoformat(),
        code_score=skill["code_score"],
        router_score=skill["router_score"],
        clarity_score=skill["clarity_score"],
        convergence_score=conv.average_score,
        optimal_path=conv.optimal_path,
        convergence_successful=conv.successful_runs,
        convergence_failed=conv.failed_runs,
        total_tokens=skill["total_tokens"],
        avg_tokens_per_case=skill["avg_tokens_per_case"],
    )
    return result, skill, conv


# ---------------------------------------------------------------------------
# Dashboard comparativo
# ---------------------------------------------------------------------------
def print_dashboard(results: list[ExperimentResult]) -> None:
    out = sys.stderr
    sep = "═" * 95

    print(f"\n{sep}", file=out)
    print(f"  DASHBOARD COMPARATIVO — {len(results)} variantes evaluadas", file=out)
    print(sep, file=out)
    print(
        f"  {'Variant':<25} | {'Code':>6} | {'Router':>7} | {'Clarity':>8} | "
        f"{'Converge':>9} | {'Tokens/caso':>11}",
        file=out,
    )
    print("─" * 95, file=out)

    for r in results:
        print(
            f"  {r.variant_name:<25} | "
            f"{r.code_score:>5.0%} | "
            f"{r.router_score:>6.0%} | "
            f"{r.clarity_score:>7.0%} | "
            f"{r.convergence_score:>8.0%} | "
            f"{r.avg_tokens_per_case:>11}",
            file=out,
        )

    print(sep, file=out)
    _print_winners(results, out, sep)


def _best_in(results: list[ExperimentResult], key: str, higher_is_better: bool = True) -> str:
    """Devuelve el nombre del variant ganador para una metrica."""
    if higher_is_better:
        winner = max(results, key=lambda r: getattr(r, key))
    else:
        winner = min(results, key=lambda r: getattr(r, key))
    return winner.variant_name


def _print_winners(results: list[ExperimentResult], out, sep: str) -> None:
    if len(results) <= 1:
        return
    print("  GANADORES POR METRICA:", file=out)
    print(f"    Mejor en code:         {_best_in(results, 'code_score')}", file=out)
    print(f"    Mejor en router:       {_best_in(results, 'router_score')}", file=out)
    print(f"    Mejor en clarity:      {_best_in(results, 'clarity_score')}", file=out)
    print(f"    Mejor en convergencia: {_best_in(results, 'convergence_score')}", file=out)
    print(f"    Mas eficiente (tokens): {_best_in(results, 'avg_tokens_per_case', higher_is_better=False)}", file=out)
    print(sep, file=out)


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------
def _convergence_to_dict(conv: ConvergenceResult) -> dict:
    """Serializa ConvergenceResult incluyendo la lista per_query."""
    return {
        "variant_name": conv.variant_name,
        "optimal_path": conv.optimal_path,
        "average_score": conv.average_score,
        "successful_runs": conv.successful_runs,
        "failed_runs": conv.failed_runs,
        "per_query": [
            {
                "query": q.query,
                "path_length": q.path_length,
                "succeeded": q.succeeded,
                "tool_called": q.tool_called,
            }
            for q in conv.per_query
        ],
    }


def save_results(
    experiments: list[ExperimentResult],
    skill_details: list[dict],
    conv_details: list[ConvergenceResult],
) -> str:
    """Guarda resultados completos en results/{timestamp}.json. Devuelve el path."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(RESULTS_DIR, f"{timestamp}.json")

    data = {
        "timestamp": timestamp,
        "summary": [asdict(e) for e in experiments],
        "skill_details": skill_details,
        "convergence_details": [_convergence_to_dict(c) for c in conv_details],
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Experiment Runner — comparando variantes ===")
    print(f"Variantes a evaluar: {[v.name for v in ALL_VARIANTS]}\n")

    experiments = []
    skill_details = []
    conv_details = []

    for variant in ALL_VARIANTS:
        print(f"\n[{ALL_VARIANTS.index(variant) + 1}/{len(ALL_VARIANTS)}] Variante: {variant.name}")
        try:
            result, skill, conv = run_experiment(variant)
            experiments.append(result)
            skill_details.append(skill)
            conv_details.append(conv)
        except Exception as e:
            print(f"  [ERROR variante {variant.name}]: {e}", file=sys.stderr)

    # Dashboard comparativo
    if experiments:
        print_dashboard(experiments)

        # Detalle de convergencia por variante (util para entender por que un score)
        for conv in conv_details:
            print_convergence_report(conv)

        # Persistir
        path = save_results(experiments, skill_details, conv_details)
        print(f"\nResultados guardados en: {path}", file=sys.stderr)
    else:
        print("\nNo se completo ningun experimento.", file=sys.stderr)


# ---------------------------------------------------------------------------
# TODO — Pendiente para version 05
# ---------------------------------------------------------------------------
# Judging the judge: meta-evaluacion del LLM-as-judge contra ground truth
# etiquetada a mano. Concepto del bonus L11. Ver memory/project_pending_topics.md
# item 8 para retomarlo.
