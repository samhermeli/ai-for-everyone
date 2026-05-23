"""
Judging the judge — meta-evaluacion del LLM-as-judge.

Concepto (L11 bonus):
  El LLM-as-judge nunca es 100% accurate. Para saber CUAN accurate es, se
  construye un dataset etiquetado a mano (ground truth) con casos donde el
  juicio correcto ya esta decidido por un humano.

  Despues le mostras los mismos casos al juez y compara su veredicto contra
  el label humano. La accuracy del juez = (casos correctos / total).

Tres niveles de evaluacion:
  Nivel 1: agente actua sobre input del usuario
  Nivel 2: judge evalua si el agente actuo bien
  Nivel 3: ESTE archivo evalua si el judge evalua bien

Caso real visto en 03_evals:
  Router judge dijo 100% accuracy. Code-based detecto 2 fallos reales del
  agente. El juez tenia sesgo positivo porque era el mismo modelo que el agente.
  Este script existe para detectar esa situacion automaticamente.

Correr:
    # Con mismo modelo (mostrara sesgo)
    export $(cat ../.env | xargs) && python mp_judge_eval.py

    # Con juez separado (mas confiable)
    export $(cat ../.env | xargs) && \\
      JUDGE_PROVIDER=groq GROQ_API_KEY=gsk_... python mp_judge_eval.py
"""

import json
import os
import sys
from dataclasses import dataclass, field

from mp_evals import (
    AgentRunData,
    JUDGE_MODEL,
    JUDGE_PROVIDER,
    eval_router_llm_judge,
)
from mp_agent import MODEL as AGENT_MODEL, PROVIDER as AGENT_PROVIDER
from judge_test_dataset import JUDGE_DATASET


# ---------------------------------------------------------------------------
# Resultado de una evaluacion del juez sobre un caso
# ---------------------------------------------------------------------------
@dataclass
class JudgeCaseResult:
    case_id: str
    user_message: str
    tool_called: str | None
    expected_label: str        # ground truth (humano)
    judge_label: str           # lo que dijo el juez
    judge_explanation: str
    judge_correct: bool        # ¿el juez coincidio con el humano?
    expected_reasoning: str


@dataclass
class JudgeEvalSummary:
    judge_model: str
    judge_provider: str
    agent_model: str
    n_cases: int
    correct: int
    incorrect: int
    accuracy: float
    # Confusion matrix sobre el label binario correct/incorrect
    true_positives: int    # juez dijo correct, GT correct
    true_negatives: int    # juez dijo incorrect, GT incorrect
    false_positives: int   # juez dijo correct, GT incorrect (FALSO ELOGIO al agente)
    false_negatives: int   # juez dijo incorrect, GT correct (FALSA ALARMA)
    cases: list[JudgeCaseResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_judge_eval() -> JudgeEvalSummary:
    """Corre el router judge sobre cada caso del dataset hand-labeled."""
    cases = []
    tp = tn = fp = fn = 0
    n = len(JUDGE_DATASET)

    for i, case in enumerate(JUDGE_DATASET, 1):
        # Construir un AgentRunData sintetico — bypassemos el agente,
        # solo queremos testear el juez.
        data = AgentRunData(
            user_message=case["user_message"],
            tool_called=case["tool_called"],
            tool_input=case["tool_input"],
        )

        print(f"     judge eval [{i:>2}/{n}] {case['id']}: {case['user_message'][:50]}...",
              flush=True)

        try:
            judge_result = eval_router_llm_judge(data)
        except Exception as e:
            print(f"  [ERROR caso {case['id']}]: {e}", file=sys.stderr)
            continue

        is_correct = (judge_result.label == case["expected_label"])

        # Confusion matrix
        if case["expected_label"] == "correct":
            if judge_result.label == "correct":
                tp += 1
            else:
                fn += 1  # judge dijo incorrect cuando era correct
        else:  # expected = incorrect
            if judge_result.label == "incorrect":
                tn += 1
            else:
                fp += 1  # judge dijo correct cuando era incorrect (peligroso)

        cases.append(JudgeCaseResult(
            case_id=case["id"],
            user_message=case["user_message"],
            tool_called=case["tool_called"],
            expected_label=case["expected_label"],
            judge_label=judge_result.label,
            judge_explanation=judge_result.explanation,
            judge_correct=is_correct,
            expected_reasoning=case["reasoning"],
        ))

    correct = sum(1 for c in cases if c.judge_correct)
    return JudgeEvalSummary(
        judge_model=JUDGE_MODEL,
        judge_provider=JUDGE_PROVIDER,
        agent_model=AGENT_MODEL,
        n_cases=len(cases),
        correct=correct,
        incorrect=len(cases) - correct,
        accuracy=correct / len(cases) if cases else 0,
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        cases=cases,
    )


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------
def print_judge_report(s: JudgeEvalSummary) -> None:
    out = sys.stderr
    sep = "═" * 80
    thin = "─" * 80

    print(f"\n{sep}", file=out)
    print("  JUDGE EVAL — accuracy del LLM-as-judge contra ground truth", file=out)
    print(sep, file=out)
    print(f"  Judge:  {s.judge_provider}/{s.judge_model}", file=out)
    print(f"  Agent:  {AGENT_PROVIDER}/{s.agent_model}", file=out)
    if s.judge_model == s.agent_model:
        print(f"  ⚠ Judge y agente son el MISMO modelo — sesgo esperado", file=out)
    print(thin, file=out)

    # Detalle por caso — mostramos los FALLOS del juez con razonamientos
    print("  Casos donde el JUEZ se equivoco:", file=out)
    any_failure = False
    for c in s.cases:
        if c.judge_correct:
            continue
        any_failure = True
        print(f"\n  [{c.case_id}] {c.user_message[:65]}", file=out)
        tool_str = c.tool_called or "ninguna"
        print(f"     Tool llamada: {tool_str}", file=out)
        print(f"     Humano dijo : {c.expected_label}  ({c.expected_reasoning})", file=out)
        print(f"     Judge dijo  : {c.judge_label}", file=out)
        if c.judge_explanation:
            exp = c.judge_explanation[:140] + "..." if len(c.judge_explanation) > 140 else c.judge_explanation
            print(f"     Judge razono: {exp}", file=out)

    if not any_failure:
        print("\n  (ninguno — el juez acerto en TODOS los casos)", file=out)

    # Confusion matrix
    print(f"\n{thin}", file=out)
    print("  Confusion matrix (sobre los labels binarios):", file=out)
    print(f"    True positives  (juez=correct, GT=correct):     {s.true_positives}", file=out)
    print(f"    True negatives  (juez=incorrect, GT=incorrect): {s.true_negatives}", file=out)
    print(f"    False positives (juez=correct, GT=incorrect):   {s.false_positives}  ← peligroso", file=out)
    print(f"    False negatives (juez=incorrect, GT=correct):   {s.false_negatives}  ← falsas alarmas", file=out)

    # Resumen
    print(f"\n{sep}", file=out)
    print(f"  RESUMEN: {s.correct}/{s.n_cases} correctos | Accuracy = {s.accuracy:.0%}", file=out)
    if s.false_positives > 0:
        print(
            f"  ⚠ El juez aprobo {s.false_positives} caso(s) donde el agente fallo — sesgo positivo presente.",
            file=out,
        )
    print(sep, file=out)


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------
def save_judge_eval(summary: JudgeEvalSummary) -> str:
    """Guarda el resultado en results/judge_eval_{timestamp}.json."""
    from datetime import datetime, timezone

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(results_dir, f"judge_eval_{timestamp}.json")

    data = {
        "timestamp": timestamp,
        "judge_provider": summary.judge_provider,
        "judge_model": summary.judge_model,
        "agent_model": summary.agent_model,
        "n_cases": summary.n_cases,
        "accuracy": summary.accuracy,
        "confusion_matrix": {
            "true_positives": summary.true_positives,
            "true_negatives": summary.true_negatives,
            "false_positives": summary.false_positives,
            "false_negatives": summary.false_negatives,
        },
        "cases": [
            {
                "case_id": c.case_id,
                "user_message": c.user_message,
                "tool_called": c.tool_called,
                "expected_label": c.expected_label,
                "judge_label": c.judge_label,
                "judge_correct": c.judge_correct,
                "judge_explanation": c.judge_explanation,
                "expected_reasoning": c.expected_reasoning,
            }
            for c in summary.cases
        ],
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\n=== Judge Eval — evaluando al router judge ===")
    print(f"Judge configurado: {JUDGE_PROVIDER}/{JUDGE_MODEL}")
    print(f"Dataset: {len(JUDGE_DATASET)} casos etiquetados a mano\n")

    summary = run_judge_eval()
    print_judge_report(summary)

    path = save_judge_eval(summary)
    print(f"\nResultados guardados en: {path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Sobre semantic similarity (no implementado, conceptual)
# ---------------------------------------------------------------------------
# Este script compara labels exactos (correct/incorrect). Si quisieras evaluar
# tambien las EXPLICACIONES del juez, necesitarias semantic similarity:
#   - Embedar la explicacion del juez y la "razon esperada" con un modelo
#     de embeddings (ej: sentence-transformers, OpenAI embeddings)
#   - Calcular cosine similarity entre los dos vectores
#   - Umbral (~0.8) para considerar que el razonamiento del juez "coincide"
#     con el humano en SIGNIFICADO aunque no sea texto identico
# Util cuando el output del juez incluye razonamiento abierto, no solo labels.
