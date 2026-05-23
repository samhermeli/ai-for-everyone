# AI for Everyone — Caso de estudio: de una API a un agente

Código fuente del artículo **"De llamar una API a construir un agente — un viaje de 5 etapas"**.

Cada carpeta es una versión progresiva del mismo agente de Mercado Pago.
Cada etapa agrega exactamente una dimensión sobre la anterior.

| Carpeta | Lo que agrega |
|---|---|
| `etapa_01/` | Agente base: LLM + tools + loop |
| `etapa_02/` | Observabilidad con OpenTelemetry |
| `etapa_03/` | Evaluadores: code-based + LLM-as-judge |
| `etapa_04/` | Experimentos: variantes + convergencia + dashboard |
| `etapa_05/` | Meta-evaluación: juzgar al juez |

---

## Requisitos

- Python 3.10+
- [Ollama](https://ollama.com) — para correr el modelo localmente
- Una cuenta de Mercado Pago con Access Token (puede ser de prueba)

---

## Setup (una sola vez)

### 1. Instalar Ollama y descargar el modelo

```bash
# Instalar Ollama (macOS)
brew install ollama

# Descargar el modelo recomendado (~5 GB)
ollama pull qwen2.5:7b

# Iniciar el servidor (dejar corriendo en una terminal)
ollama serve
```

> El modelo `qwen2.5:7b` funciona en cualquier Mac con 8 GB de RAM o más.
> Si tienes más de 16 GB, puedes usar `qwen2.5:14b` para mejor calidad.

### 2. Crear el archivo `.env`

Crea un archivo `.env` en la raíz del proyecto con este contenido:

```
MP_ACCESS_TOKEN=APP_USR-tu-token-de-mercadopago
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:7b
```

- `MP_ACCESS_TOKEN`: tu Access Token de Mercado Pago. Puede ser un token de prueba
  (comienza con `APP_USR-`). Se obtiene en [mercadopago.com/developers](https://www.mercadopago.com/developers).
- `LLM_PROVIDER`: usa `ollama` para correr localmente. Alternativas: `groq` (free tier),
  `openai` (de pago).
- `OLLAMA_MODEL`: el modelo que descargaste.

### 3. Crear entorno virtual e instalar dependencias

```bash
python3 -m venv .venv
source .venv/bin/activate

# Instalar las dependencias de la etapa que quieras correr
pip install -r etapa_01/requirements.txt
```

> El `requirements.txt` es el mismo para todas las etapas. Instalar una vez alcanza.

---

## Cómo correr cada etapa

En todos los casos, primero carga las variables de entorno:

```bash
export $(cat .env | xargs)
```

### Etapa 01 — Agente base

```bash
python etapa_01/agent_mercadopago.py
```

**Prueba con:** `Crea un link de pago para unas zapatillas a 1000 pesos`

---

### Etapa 02 — Con tracing

```bash
python etapa_02/agent_mercadopago_traced.py
```

Las trazas se imprimen en la terminal al terminar cada respuesta.
Para guardarlas en un archivo:

```bash
python etapa_02/agent_mercadopago_traced.py 2>trace.log
tail -f trace.log  # en otra terminal
```

---

### Etapa 03 — Con evaluadores

El agente y los evaluadores están separados en dos archivos:

```bash
# Solo el agente (como en producción)
python etapa_03/mp_agent.py

# Suite de evaluaciones (desarrollo / CI)
python etapa_03/mp_evals.py
```

---

### Etapa 04 — Experimentos y comparación de variantes

```bash
# Solo el agente (variante baseline)
python etapa_04/mp_agent.py

# Convergence eval sobre la variante baseline
python etapa_04/mp_convergence.py

# Experimento completo: 3 variantes × todos los evaluadores
# Tarda ~20 minutos con qwen2.5:7b
python etapa_04/mp_experiments.py 2>experiments.log
```

---

### Etapa 05 — Meta-evaluación del juez

```bash
# Agente con juez separado (mismo modelo por defecto)
python etapa_05/mp_agent.py

# Meta-evaluación: mide la accuracy del LLM-as-judge
python etapa_05/mp_judge_eval.py
```

Para usar un juez independiente (recomendado — requiere cuenta gratuita en Groq):

```bash
export JUDGE_PROVIDER=groq
export GROQ_API_KEY=gsk_tu_api_key
python etapa_05/mp_judge_eval.py
```

---

## Proveedores de LLM alternativos

Si no quieres correr Ollama localmente, puedes usar Groq (free tier):

1. Crea una cuenta en [console.groq.com](https://console.groq.com)
2. Genera un API key
3. Actualiza el `.env`:

```
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_tu_api_key
```

No se necesita cambiar nada en el código.
