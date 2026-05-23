"""
Version 02: Agente Mercado Pago con Tracing (Observabilidad).

Basado en 01_simple/python_raw/agent_mercadopago.py.
Usa OpenTelemetry puro sin UI externa.

Los spans se imprimen como arbol en la terminal al final de cada ejecucion.

Jerarquia de spans:
    AgentRun (agent)
    └── router_call_N (chain)      una por iteracion del while True
        ├── [LLM call automatico]  capturado por OpenAIInstrumentor
        └── handle_tool_calls (chain)
            └── create_payment_link (tool)
                └── mp_api_call (chain)

Setup:
    pip install -r 02_tracing/requirements.txt

Correr:
    export $(cat .env | xargs) && python 02_tracing/agent_mercadopago_traced.py
"""

import json
import os
import sys
from collections import defaultdict

import requests
from openai import OpenAI
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode


# Constante para evitar literales duplicados (SonarQube S1192)
SPAN_KIND = "openinference.span.kind"

# ---------------------------------------------------------------------------
# Setup OpenTelemetry con InMemoryExporter
# InMemorySpanExporter guarda spans en memoria — cero dependencias externas.
# ---------------------------------------------------------------------------
memory_exporter = InMemorySpanExporter()
provider = TracerProvider(resource=Resource.create({"service.name": "mercadopago-agent"}))
provider.add_span_processor(SimpleSpanProcessor(memory_exporter))
trace.set_tracer_provider(provider)

# Captura automatica de todas las llamadas al cliente openai (tokens, modelo, input/output)
OpenAIInstrumentor().instrument()

tracer = trace.get_tracer(__name__)


# ---------------------------------------------------------------------------
# Utilidad: imprimir spans como arbol en terminal
# ---------------------------------------------------------------------------
def _build_span_tree(spans):
    """Construye mapa parent->hijos y lista de raices."""
    span_ids = {s.context.span_id for s in spans}
    span_map = {s.context.span_id: s for s in spans}
    children = defaultdict(list)
    roots = []

    for span in spans:
        pid = span.parent.span_id if span.parent else None
        if pid and pid in span_ids:
            children[pid].append(span.context.span_id)
        else:
            roots.append(span.context.span_id)

    return span_map, children, roots




def print_trace(title: str = ""):
    spans = memory_exporter.get_finished_spans()
    if not spans:
        print("  (sin spans registrados)")
        return

    span_map, children, roots = _build_span_tree(spans)
    sep = "─" * 55

    # Escribir la traza a stderr — separado del output al usuario que va a stdout.
    # En produccion esto iria a un archivo de log o a un backend de observabilidad.
    # El usuario final que consume stdout nunca ve esto.
    out = sys.stderr

    print(f"\n{sep}", file=out)
    if title:
        print(f"  Traza: {title}", file=out)
    print(sep, file=out)
    for root_id in roots:
        _print_span_recursive_detailed(root_id, span_map, children, out=out)

    # Resumen de tokens — suma todos los LLM calls de la traza
    total_prompt = sum(
        s.attributes.get("llm.token_count.prompt", 0) for s in spans
    )
    total_completion = sum(
        s.attributes.get("llm.token_count.completion", 0) for s in spans
    )
    total = total_prompt + total_completion
    if total > 0:
        print(sep, file=out)
        print(f"  Tokens — prompt: {total_prompt} | completion: {total_completion} | total: {total}", file=out)

    print(sep, file=out)
    memory_exporter.clear()


def _print_span_recursive_detailed(sid, span_map, children, indent=0, out=None):
    out = out or sys.stderr

    span = span_map[sid]
    ms = (span.end_time - span.start_time) / 1_000_000
    icon = "✗" if span.status.status_code == StatusCode.ERROR else "✓"
    prefix = "    " * indent + ("└── " if indent > 0 else "")
    kind = span.attributes.get(SPAN_KIND, "")
    kind_tag = f" [{kind}]" if kind else ""
    print(f"{prefix}{icon} {span.name}{kind_tag}  ({ms:.0f}ms)", file=out)

    # Imprimir atributos relevantes con indentacion
    attr_prefix = "    " * (indent + 1) + "   "

    # DEBUG: descomentar para ver TODOS los atributos del span y descubrir nombres reales
    # for k, v in span.attributes.items():
    #     print(f"{attr_prefix}[DEBUG] {k}: {v}", file=out)

    interesting = [
        # Agente
        "user.message", "agent.response",
        # Router
        "iteration", "tool_calls.names", "response",
        # Tool
        "tool.input", "tool.output",
        # HTTP
        "http.url", "http.body", "http.status_code",
        # Tokens — nombres estandar de OpenInference para LLM calls
        "llm.token_count.prompt",
        "llm.token_count.completion",
        "llm.token_count.total",
        # Modelo usado
        "llm.model_name",
    ]
    for key in interesting:
        val = span.attributes.get(key)
        if val:
            val_str = str(val)
            if len(val_str) > 120:
                val_str = val_str[:117] + "..."
            print(f"{attr_prefix}· {key}: {val_str}", file=out)

    for child_id in children.get(sid, []):
        _print_span_recursive_detailed(child_id, span_map, children, indent + 1, out)


# ---------------------------------------------------------------------------
# Cliente LLM
# ---------------------------------------------------------------------------
PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

if PROVIDER == "ollama":
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
elif PROVIDER == "groq":
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
    )
    MODEL = "llama-3.3-70b-versatile"
elif PROVIDER == "openai":
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    MODEL = "gpt-4o-mini"
else:
    raise ValueError(f"LLM_PROVIDER invalido: {PROVIDER}")

print(f"Provider={PROVIDER} | Modelo={MODEL}")

MP_API_URL = "https://api.mercadopago.com/checkout/preferences"
MP_ACCESS_TOKEN = os.environ.get("MP_ACCESS_TOKEN")
if not MP_ACCESS_TOKEN:
    raise RuntimeError("Falta MP_ACCESS_TOKEN.")


# ---------------------------------------------------------------------------
# Tool: HTTP a MP separado para reducir complejidad cognitiva (SonarQube S3776)
# ---------------------------------------------------------------------------
def _call_mp_api(payload: dict, headers: dict) -> tuple[str, bool]:
    """Ejecuta el POST a MP. Devuelve (result_json, success)."""
    with tracer.start_as_current_span("mp_api_call") as span:
        span.set_attribute(SPAN_KIND, "chain")
        span.set_attribute("http.url", MP_API_URL)
        span.set_attribute("http.body", json.dumps(payload))
        try:
            response = requests.post(MP_API_URL, json=payload, headers=headers, timeout=30)
            span.set_attribute("http.status_code", response.status_code)
            if response.status_code in (200, 201):
                data = response.json()
                result = json.dumps({
                    "id": data.get("id"),
                    "init_point": data.get("init_point"),
                    "sandbox_init_point": data.get("sandbox_init_point"),
                })
                span.set_status(StatusCode.OK)
                return result, True
            error = json.dumps({"error": f"MP API {response.status_code}", "body": response.text})
            span.set_status(StatusCode.ERROR, f"HTTP {response.status_code}")
            return error, False
        except requests.RequestException as e:
            error = json.dumps({"error": str(e)})
            span.set_status(StatusCode.ERROR, str(e))
            return error, False


def create_payment_link(title: str, quantity: int, unit_price: float) -> str:
    """Crea un link de pago en Mercado Pago."""
    payload = {"items": [{"title": title, "quantity": quantity, "unit_price": unit_price}]}
    headers = {"Authorization": f"Bearer {MP_ACCESS_TOKEN}", "Content-Type": "application/json"}

    with tracer.start_as_current_span("create_payment_link") as span:
        span.set_attribute(SPAN_KIND, "tool")
        span.set_attribute("tool.input", json.dumps({"title": title, "quantity": quantity, "unit_price": unit_price}))

        result, success = _call_mp_api(payload, headers)

        span.set_attribute("tool.output", result)
        span.set_status(StatusCode.OK if success else StatusCode.ERROR)
        return result


# ---------------------------------------------------------------------------
# Tool schema y router
# ---------------------------------------------------------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "create_payment_link",
            "description": "Crea un link de pago en Mercado Pago.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Nombre del producto."},
                    "quantity": {"type": "integer", "description": "Cantidad de unidades."},
                    "unit_price": {"type": "number", "description": "Precio unitario."},
                },
                "required": ["title", "quantity", "unit_price"],
            },
        },
    }
]

tool_implementations = {"create_payment_link": create_payment_link}

SYSTEM_PROMPT = """
Eres un asistente que crea links de pago en Mercado Pago.

Tu unico trabajo es recopilar tres datos y llamar create_payment_link:
  - titulo: nombre del producto
  - quantity: cantidad de unidades (si no se menciona, asume 1)
  - unit_price: precio unitario en numeros

Reglas estrictas:
  1. Revisa TODO el historial de la conversacion antes de preguntar algo.
     Si el usuario ya dio un dato antes, NO lo pidas de nuevo.
  2. Si falta un dato, pregunta SOLO ese dato. Una pregunta a la vez.
  3. Si el usuario da cantidad implicita ("quiero vender uno", "un zapato"),
     asume quantity=1 sin preguntar.
  4. Cuando tengas los tres datos, llama la tool de inmediato sin confirmar.
  5. Responde siempre en el mismo idioma que el usuario.
  6. Al entregar el link, muestra solo el sandbox_init_point con un mensaje simple.
     No expliques detalles tecnicos al usuario.
"""


def handle_tool_calls(tool_calls, messages):
    with tracer.start_as_current_span("handle_tool_calls") as span:
        span.set_attribute(SPAN_KIND, "chain")
        span.set_attribute("tool_calls.count", len(tool_calls))
        for tool_call in tool_calls:
            fn = tool_implementations[tool_call.function.name]
            args = json.loads(tool_call.function.arguments)
            result = fn(**args)
            messages.append({"role": "tool", "content": result, "tool_call_id": tool_call.id})
        span.set_status(StatusCode.OK)
    return messages


def run_agent(messages):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if not any(isinstance(m, dict) and m.get("role") == "system" for m in messages):
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    iteration = 0
    while True:
        iteration += 1
        with tracer.start_as_current_span(f"router_call_{iteration}") as span:
            span.set_attribute(SPAN_KIND, "chain")
            span.set_attribute("iteration", iteration)

            response = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
            msg = response.choices[0].message
            messages.append(msg.model_dump())
            tool_calls = msg.tool_calls

            if tool_calls:
                span.set_attribute("tool_calls.names", str([tc.function.name for tc in tool_calls]))
                span.set_status(StatusCode.OK)
                messages = handle_tool_calls(tool_calls, messages)
            else:
                span.set_attribute("response", msg.content or "")
                span.set_status(StatusCode.OK)
                return msg.content


def start_main_span(user_message: str) -> str:
    with tracer.start_as_current_span("AgentRun") as span:
        span.set_attribute(SPAN_KIND, "agent")
        span.set_attribute("user.message", user_message)
        try:
            result = run_agent(user_message)
            span.set_attribute("agent.response", result or "")
            span.set_status(StatusCode.OK)
            return result
        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            raise


# ---------------------------------------------------------------------------
# Chat interactivo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Agente MP con Tracing (OpenTelemetry) ====")
    print("Escribi tu pedido o 'salir' para terminar.")
    print("(Las trazas van a stderr — corré con '2>trace.log' para separarlas)\n")

    while True:
        user_input = input("Tu: ").strip()
        if user_input.lower() in ("salir", "exit", "quit"):
            break
        if not user_input:
            continue
        try:
            result = start_main_span(user_input)

            # stdout → lo que ve el usuario final
            print(f"\nAgente: {result}\n")

            # stderr → las trazas, invisibles para el usuario final
            print_trace(title=user_input[:50])

        except Exception as e:
            print(f"\n[error] {e}\n")
