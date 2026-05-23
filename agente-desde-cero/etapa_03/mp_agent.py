"""
Agente Mercado Pago — codigo de producto.

Este archivo es lo que iria a produccion. Contiene:
  - El agente (run_agent, start_main_span)
  - Su tool (create_payment_link)
  - El system prompt
  - Configuracion de tracing (OpenTelemetry)

NO contiene evals, datasets de prueba ni evaluadores.
Esos viven en mp_evals.py, que importa de este archivo.

Se puede correr de forma standalone (chat interactivo):
    export $(cat ../.env | xargs) && python mp_agent.py
"""

import json
import os
import sys

import requests
from openai import OpenAI
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode


# Constante para evitar literales duplicados
SPAN_KIND = "openinference.span.kind"

# ---------------------------------------------------------------------------
# Setup OpenTelemetry
# memory_exporter se exporta para que mp_evals pueda leer los spans del agente
# ---------------------------------------------------------------------------
memory_exporter = InMemorySpanExporter()
provider = TracerProvider(resource=Resource.create({"service.name": "mercadopago-agent"}))
provider.add_span_processor(SimpleSpanProcessor(memory_exporter))
trace.set_tracer_provider(provider)
OpenAIInstrumentor().instrument()
tracer = trace.get_tracer(__name__)


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
# Tool: create_payment_link
# ---------------------------------------------------------------------------
def _call_mp_api(payload: dict, headers: dict) -> tuple[str, bool]:
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
        span.set_attribute("tool.input", json.dumps({
            "title": title, "quantity": quantity, "unit_price": unit_price
        }))
        result, success = _call_mp_api(payload, headers)
        span.set_attribute("tool.output", result)
        span.set_status(StatusCode.OK if success else StatusCode.ERROR)
        return result


# ---------------------------------------------------------------------------
# Schema de la tool y router
# ---------------------------------------------------------------------------
tools_schema = [
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
  1. Revisa TODO el historial antes de preguntar algo.
     Si el usuario ya dio un dato, NO lo pidas de nuevo.
  2. Si falta un dato, pregunta SOLO ese. Una pregunta a la vez.
  3. Si el usuario da cantidad implicita, asume quantity=1 sin preguntar.
  4. Cuando tengas los tres datos, llama la tool de inmediato.
  5. Responde siempre en el mismo idioma que el usuario.
  6. Al entregar el link, muestra solo el sandbox_init_point con un mensaje simple.
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


def run_agent(user_message: str, max_iterations: int = 10) -> str:
    """
    Loop del router con limite de iteraciones (circuit breaker).
    Si el LLM sigue llamando tools sin nunca devolver respuesta final,
    corta despues de max_iterations para no consumir tokens infinitamente.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    for iteration in range(1, max_iterations + 1):
        with tracer.start_as_current_span(f"router_call_{iteration}") as span:
            span.set_attribute(SPAN_KIND, "chain")
            span.set_attribute("iteration", iteration)
            response = client.chat.completions.create(
                model=MODEL, messages=messages, tools=tools_schema
            )
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
                return msg.content or ""

    raise RuntimeError(
        f"El agente excedio {max_iterations} iteraciones sin devolver respuesta final"
    )


def start_main_span(user_message: str) -> str:
    """Punto de entrada con span raiz tipo agent."""
    with tracer.start_as_current_span("AgentRun") as span:
        span.set_attribute(SPAN_KIND, "agent")
        span.set_attribute("user.message", user_message)
        try:
            result = run_agent(user_message)
            span.set_attribute("agent.response", result)
            span.set_status(StatusCode.OK)
            return result
        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            raise


# ---------------------------------------------------------------------------
# Modo standalone — chat interactivo
# Cuando se corre directamente este archivo (no como import desde evals)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Agente Mercado Pago (standalone) ===")
    print("Escribi tu pedido o 'salir' para terminar.\n")

    while True:
        user_input = input("Tu: ").strip()
        if user_input.lower() in ("salir", "exit", "quit"):
            break
        if not user_input:
            continue
        try:
            result = start_main_span(user_input)
            print(f"\nAgente: {result}\n")
            # Limpiar spans para no acumular en memoria entre turnos
            memory_exporter.clear()
        except Exception as e:
            print(f"\n[error] {e}\n", file=sys.stderr)
