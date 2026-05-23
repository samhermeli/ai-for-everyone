"""
Agente Python con UNA sola tool: crear links de pago en Mercado Pago.

Disenado para probar localmente sin pagar API key de OpenAI.
Soporta tres providers via variable de entorno LLM_PROVIDER:

  - "ollama"  -> corre 100% local (gratis). Requiere instalar Ollama y
                un modelo con soporte de tools (ej: llama3.1, qwen2.5).
                  brew install ollama  (o desde ollama.com)
                  ollama pull llama3.1
                  ollama serve   # corre en http://localhost:11434

  - "groq"    -> free tier en la nube (gratis con cuenta). Muy rapido.
                  Pedi API key en https://console.groq.com
                  Exporta: GROQ_API_KEY=gsk_...

  - "openai"  -> oficial, de pago. Exporta OPENAI_API_KEY=sk-...

Setup minimo:
  pip install openai requests python-dotenv

Variables de entorno requeridas:
  LLM_PROVIDER=ollama   (o groq, openai)
  MP_ACCESS_TOKEN=APP_USR-...     # tu token de Mercado Pago
  GROQ_API_KEY=gsk_...            # solo si usas groq
  OPENAI_API_KEY=sk-...           # solo si usas openai

Ejecutar:
  python agent_mercadopago.py
"""

import json
import os

import requests
from openai import OpenAI


# ---------------------------------------------------------------------------
# Configuracion del cliente segun el provider elegido
# ---------------------------------------------------------------------------
PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

if PROVIDER == "ollama":
    # Ollama expone API compatible con OpenAI en localhost.
    # No requiere API key real (cualquier string sirve).
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # placeholder, Ollama no valida la key
    )
    # qwen2.5 tiene excelente soporte de function calling.
    # Cambia el tag segun el tamano que quieras: 7b, 14b, 32b.
    MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")

elif PROVIDER == "groq":
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
    )
    MODEL = "llama-3.3-70b-versatile"  # gratis en Groq free tier

elif PROVIDER == "openai":
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    MODEL = "gpt-4o-mini"

else:
    raise ValueError(f"LLM_PROVIDER invalido: {PROVIDER}")

print(f"Usando provider={PROVIDER} con modelo={MODEL}")


# ---------------------------------------------------------------------------
# Tool unica: crear preferencia de pago en Mercado Pago
# ---------------------------------------------------------------------------
MP_API_URL = "https://api.mercadopago.com/checkout/preferences"
MP_ACCESS_TOKEN = os.environ.get("MP_ACCESS_TOKEN")

if not MP_ACCESS_TOKEN:
    raise RuntimeError(
        "Falta MP_ACCESS_TOKEN. Exportalo: export MP_ACCESS_TOKEN=APP_USR-..."
    )


def create_payment_link(title: str, quantity: int, unit_price: float) -> str:
    """
    Crea una preferencia de pago en Mercado Pago y devuelve el link (init_point).

    Esta es la version minima que pediste (solo title, quantity, unit_price).
    El resto de campos del payload original son opcionales para MP.
    """
    payload = {
        "items": [
            {
                "title": title,
                "quantity": quantity,
                "unit_price": unit_price,
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {MP_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(MP_API_URL, json=payload, headers=headers, timeout=30)
    except requests.RequestException as e:
        return json.dumps({"error": f"Network error: {str(e)}"})

    # Devolvemos JSON serializable para que el LLM lo pueda leer.
    if response.status_code in (200, 201):
        data = response.json()
        return json.dumps({
            "id": data.get("id"),
            "init_point": data.get("init_point"),            # link de pago "produccion"
            "sandbox_init_point": data.get("sandbox_init_point"),  # link sandbox
            "date_created": data.get("date_created"),
        })
    else:
        return json.dumps({
            "error": f"MP API returned {response.status_code}",
            "body": response.text,
        })


# ---------------------------------------------------------------------------
# Schema de la tool en formato OpenAI function calling
# ---------------------------------------------------------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "create_payment_link",
            "description": (
                "Crea un link de pago en Mercado Pago. "
                "Usalo cuando el usuario pida generar un cobro, crear un link de pago, "
                "o vender un producto online."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Nombre del producto o servicio a cobrar.",
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Cantidad de unidades. Default 1 si el usuario no lo especifica.",
                    },
                    "unit_price": {
                        "type": "number",
                        "description": "Precio unitario en la moneda local (sin centavos, ej: 1500 = $1500).",
                    },
                },
                "required": ["title", "quantity", "unit_price"],
            },
        },
    }
]

tool_implementations = {
    "create_payment_link": create_payment_link,
}


# ---------------------------------------------------------------------------
# Router: mismo patron que el agente original
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
Eres un asistente que ayuda a vendedores a crear links de pago en Mercado Pago.

Cuando el usuario te pida cobrar algo o crear un link de pago:
1. Identifica el producto, la cantidad y el precio.
2. Si falta alguno de esos datos, preguntale al usuario antes de llamar la tool.
3. Una vez que tengas los tres datos, llama a create_payment_link.
4. Devuelve al usuario el init_point (o sandbox_init_point) en formato amigable.
"""


def handle_tool_calls(tool_calls, messages):
    for tool_call in tool_calls:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        print(f"  -> Ejecutando {name}({args})")

        function = tool_implementations[name]
        result = function(**args)

        messages.append({
            "role": "tool",
            "content": result,
            "tool_call_id": tool_call.id,
        })
    return messages


def run_agent(user_message: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    while True:
        print("\n[router] llamando al LLM...")
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
        )

        msg = response.choices[0].message
        messages.append(msg)

        tool_calls = msg.tool_calls
        if tool_calls:
            print(f"[router] LLM pide {len(tool_calls)} tool call(s)")
            messages = handle_tool_calls(tool_calls, messages)
        else:
            print("[router] sin tool calls, devolviendo respuesta final")
            return msg.content or ""


# ---------------------------------------------------------------------------
# Modo interactivo: chat por consola
# ---------------------------------------------------------------------------
def chat_loop():
    print("\n=== Agente Mercado Pago ===")
    print("Escribi tu pedido (ej: 'cobrame 1500 pesos por unas zapatillas')")
    print("Escribi 'salir' para terminar.\n")

    while True:
        user_input = input("Tu: ").strip()
        if user_input.lower() in ("salir", "exit", "quit"):
            break
        if not user_input:
            continue

        try:
            result = run_agent(user_input)
            print(f"\nAgente: {result}\n")
        except Exception as e:
            print(f"\n[error] {e}\n")


if __name__ == "__main__":
    # Opcion 1: modo interactivo
    chat_loop()

    # Opcion 2: pregunta unica (descomentar para probar rapido)
    # result = run_agent("Crea un link de pago para vender unas zapatillas Meli Sport a 1500 pesos, 1 unidad")
    # print(result)
