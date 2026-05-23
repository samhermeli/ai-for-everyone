"""
Convergence eval — mide si el agente toma el camino optimo cuando recibe
preguntas equivalentes (paraphrases del mismo intent).

Concepto (L9):
  - Trayectoria = el camino que toma el agente (router → tool → router → ...)
  - Path length = numero de iteraciones del router (spans router_call_N)
  - Optimal path = el minimo de path_length entre las queries exitosas
  - Convergence score = optimal_path / actual_path (entre 0 y 1)
  - Score = 1.0 → el agente siempre toma el camino mas corto

Limitacion (L9):
  Convergence NO detecta pasos innecesarios que el agente hace SIEMPRE.
  Si todas las queries toman 5 pasos pero el optimo seria 3, el score
  sera 1.0 igual porque optimal = min(actuals) = 5.

Standalone:
    export $(cat ../.env | xargs) && python mp_convergence.py
"""

import sys
from dataclasses import dataclass, field
from statistics import mean

from mp_agent import (
    AgentVariant,
    VARIANT_BASELINE,
    memory_exporter,
    start_main_span,
)


# ---------------------------------------------------------------------------
# Dataset — 12 paraphrases del MISMO intent:
# "vender un zapato a 500 pesos, 1 unidad"
#
# Todas deberian generar la misma trayectoria optima: router → tool → router
# (2 iteraciones del router: una para llamar la tool, otra para redactar respuesta).
#
# Si alguna variante del prompt confunde al modelo con cierta formulacion,
# va a tomar mas iteraciones — la convergencia lo detecta.
# ---------------------------------------------------------------------------
CONVERGENCE_DATASET = [
    "Vendo un zapato a 500 pesos, 1 unidad",
    "Cobrame un zapato por 500 pesos",
    "Necesito un link para vender un zapato a 500",
    "Crea un link de pago para zapato de 500",
    "Genera un cobro de zapato a 500",
    "Link de pago: zapato, 500",
    "Quiero cobrar 500 por un zapato",
    "Hacer un cobro de 500 pesos para zapato",
    "Vendo zapato. Precio 500.",
    "Cobrame 500 por un zapato",
    "Genera link de pago zapato 500",
    "Necesito vender un zapato a 500 pesos",
]


# ---------------------------------------------------------------------------
# Resultado de convergencia
# ---------------------------------------------------------------------------
@dataclass
class ConvergenceQueryResult:
    query: str
    path_length: int      # 0 si fallo
    succeeded: bool       # si el agente respondio sin error
    tool_called: bool     # si llamo create_payment_link


@dataclass
class ConvergenceResult:
    variant_name: str
    optimal_path: int
    average_score: float
    successful_runs: int
    failed_runs: int
    per_query: list = field(default_factory=list)


def _count_router_calls(spans) -> int:
    """Cuenta spans cuyo nombre empieza con 'router_call_' — eso ES el path length."""
    return sum(1 for s in spans if s.name.startswith("router_call_"))


def _tool_was_called(spans) -> bool:
    return any(s.name == "create_payment_link" for s in spans)


def evaluate_convergence(variant: AgentVariant) -> ConvergenceResult:
    """
    Corre cada query del dataset con la variante dada.
    Calcula optimal_path = min(path_length de runs exitosos).
    Calcula score por query = optimal / path_length.
    """
    per_query = []
    n = len(CONVERGENCE_DATASET)

    for i, query in enumerate(CONVERGENCE_DATASET, 1):
        # Progreso a stdout — visible siempre
        print(f"     conv  [{i:>2}/{n}] {query[:60]}...", flush=True)
        memory_exporter.clear()
        succeeded = True

        try:
            start_main_span(query, variant=variant)
        except Exception:
            # Lab 9 dice: solo contar runs exitosos para optimal_path.
            # Pero registramos el fallo en per_query para visibilidad.
            succeeded = False

        spans = memory_exporter.get_finished_spans()
        path_length = _count_router_calls(spans) if succeeded else 0
        tool_called = _tool_was_called(spans) if succeeded else False

        per_query.append(ConvergenceQueryResult(
            query=query,
            path_length=path_length,
            succeeded=succeeded,
            tool_called=tool_called,
        ))
        memory_exporter.clear()

    # Optimal solo sobre runs exitosos (limitacion documentada en L9)
    successful_lengths = [r.path_length for r in per_query if r.succeeded and r.path_length > 0]
    if not successful_lengths:
        return ConvergenceResult(
            variant_name=variant.name,
            optimal_path=0,
            average_score=0.0,
            successful_runs=0,
            failed_runs=len(per_query),
            per_query=per_query,
        )

    optimal_path = min(successful_lengths)
    scores = [optimal_path / r.path_length for r in per_query
              if r.succeeded and r.path_length > 0]

    return ConvergenceResult(
        variant_name=variant.name,
        optimal_path=optimal_path,
        average_score=mean(scores) if scores else 0.0,
        successful_runs=len(successful_lengths),
        failed_runs=len(per_query) - len(successful_lengths),
        per_query=per_query,
    )


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------
def print_convergence_report(result: ConvergenceResult, out=sys.stderr) -> None:
    sep = "═" * 75
    thin = "─" * 75

    print(f"\n{sep}", file=out)
    print(f"  CONVERGENCE — variante: {result.variant_name}", file=out)
    print(sep, file=out)
    print(
        f"  Successful runs: {result.successful_runs}/{result.successful_runs + result.failed_runs} | "
        f"Optimal path: {result.optimal_path} iteraciones | "
        f"Score promedio: {result.average_score:.0%}",
        file=out,
    )
    print(thin, file=out)
    print(f"  {'Query':<55} {'Path':>5} {'Score':>7}  Tool?", file=out)
    print(thin, file=out)

    for r in result.per_query:
        if r.succeeded and r.path_length > 0:
            score = result.optimal_path / r.path_length if result.optimal_path else 0
            score_str = f"{score:.0%}"
            tool_mark = "✓" if r.tool_called else "✗"
        else:
            score_str = "-"
            tool_mark = "ERR"

        query_short = r.query[:53] + ".." if len(r.query) > 55 else r.query
        path_str = str(r.path_length) if r.succeeded else "ERR"
        print(f"  {query_short:<55} {path_str:>5} {score_str:>7}  {tool_mark}", file=out)

    print(sep, file=out)


# ---------------------------------------------------------------------------
# Standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Convergence Eval — solo variante baseline ===")
    print(f"Ejecutando {len(CONVERGENCE_DATASET)} queries paraphrase...\n")

    result = evaluate_convergence(VARIANT_BASELINE)

    print("\nCompletado.")
    print_convergence_report(result)
