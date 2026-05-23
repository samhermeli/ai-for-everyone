# AI for Everyone

> Código fuente de artículos sobre inteligencia artificial aplicada — explicada paso a paso, con código que se puede correr.

![Python](https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white)
![LLM](https://img.shields.io/badge/LLM-Ollama%20%7C%20Groq%20%7C%20OpenAI-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Sobre este repositorio

Cada directorio es un artículo independiente con su propio README de setup, dependencias y comandos. La idea es ir de lo simple a lo complejo, sumando una sola dimensión por etapa, para que se entienda **el porqué de cada pieza** y no solo el cómo.

## Artículos disponibles

| Artículo | Etapas | Carpeta |
|---|---|---|
| **De llamar una API a construir un agente** — un viaje de 5 etapas: del primer LLM call hasta meta-evaluación del juez. | 5 | [`agente-desde-cero/`](./agente-desde-cero) |

> Más artículos en camino.

## Estructura

```
ai-for-everyone/
├── README.md                  ← estás aquí
└── agente-desde-cero/         ← un artículo
    ├── README.md              ← setup específico del artículo
    ├── etapa_01/              ← una idea por etapa
    ├── etapa_02/
    └── ...
```

## Cómo empezar

1. Cloná el repo:
   ```bash
   git clone https://github.com/samhermeli/ai-for-everyone.git
   ```
2. Entrá a la carpeta del artículo que te interese.
3. Seguí las instrucciones del `README.md` de esa carpeta.

Cada artículo es autocontenido — no hace falta correr los anteriores para entender el siguiente.

## Stack que vas a ver

- **Python 3.10+** como lenguaje base.
- **Ollama / Groq / OpenAI** como proveedores de LLM (intercambiables).
- **OpenTelemetry** para observabilidad.
- **LLM-as-judge** y **convergence evaluation** para medir calidad.

## Contribuciones

Este repo es un acompañamiento de artículos, no una librería. Si encontrás un bug o algo que se puede explicar mejor, abrí un issue.

## Licencia

MIT
