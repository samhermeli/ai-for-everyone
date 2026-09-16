#!/usr/bin/env python3
"""Prueba de concepto: SecureBERT 2.0 para clasificar métricas CVSS 4.0.

El archivo es autocontenido y ofrece tres comandos:

    python securebert_cvss4_demo.py inspect
    python securebert_cvss4_demo.py train --quick
    python securebert_cvss4_demo.py predict --text "Remote attackers ..."

`inspect` solo usa la biblioteca estándar. `train` y `predict` requieren
PyTorch y Transformers. El entrenamiento produce un checkpoint al lado del
script; no modifica el dataset ni el proyecto de la demo.

Esta es una prueba de concepto educativa. Los umbrales y las confianzas no
deben utilizarse en producción sin evaluación, calibración y revisión humana.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


MODEL_ID = "cisco-ai/SecureBERT2.0-base"
DATASET_NAME = "nvdcve-2.0-2025.json.gz"
CHECKPOINT_NAME = "securebert_cvss4_demo.pt"

# Las once métricas Base de CVSS 4.0. La confianza del encoder no forma parte
# del vector: solo ayuda al gate a decidir si acepta o escala al fallback.
METRIC_VALUES: dict[str, tuple[str, ...]] = {
    "AV": ("N", "A", "L", "P"),
    "AC": ("L", "H"),
    "AT": ("N", "P"),
    "PR": ("N", "L", "H"),
    "UI": ("N", "P", "A"),
    "VC": ("H", "L", "N"),
    "VI": ("H", "L", "N"),
    "VA": ("H", "L", "N"),
    "SC": ("H", "L", "N"),
    "SI": ("H", "L", "N"),
    "SA": ("H", "L", "N"),
}
METRICS = tuple(METRIC_VALUES)
VALUE_TO_ID = {
    metric: {value: index for index, value in enumerate(values)}
    for metric, values in METRIC_VALUES.items()
}


@dataclass(frozen=True)
class CvssRecord:
    cve_id: str
    description: str
    labels: tuple[int, ...]
    vector: str
    source_type: str


def default_dataset_path() -> Path:
    script_directory = Path(__file__).resolve().parent
    alongside_script = script_directory / DATASET_NAME
    if alongside_script.is_file():
        return alongside_script
    return script_directory.parent / DATASET_NAME


def default_checkpoint_path() -> Path:
    return Path(__file__).resolve().with_name(CHECKPOINT_NAME)


def parse_base_vector(vector: str) -> dict[str, str] | None:
    """Extrae las once métricas Base y rechaza vectores incompletos."""
    if not vector.startswith("CVSS:4.0/"):
        return None
    values: dict[str, str] = {}
    for component in vector.split("/")[1:]:
        if ":" not in component:
            continue
        metric, value = component.split(":", 1)
        if metric in METRIC_VALUES:
            values[metric] = value
    if set(values) != set(METRICS):
        return None
    if any(values[metric] not in VALUE_TO_ID[metric] for metric in METRICS):
        return None
    return values


def english_description(cve: dict[str, Any]) -> str | None:
    for item in cve.get("descriptions", []):
        if item.get("lang") == "en" and isinstance(item.get("value"), str):
            description = " ".join(item["value"].split())
            if description:
                return description
    return None


def load_records(path: Path) -> tuple[list[CvssRecord], dict[str, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset no encontrado: {path}")

    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)

    vulnerabilities = payload.get("vulnerabilities")
    if not isinstance(vulnerabilities, list):
        raise ValueError("El JSON no tiene la lista `vulnerabilities` esperada.")

    records: list[CvssRecord] = []
    stats = Counter(total_cves=len(vulnerabilities))
    seen_ids: set[str] = set()

    for wrapper in vulnerabilities:
        cve = wrapper.get("cve", {})
        cve_id = cve.get("id")
        if not isinstance(cve_id, str) or not cve_id:
            stats["missing_id"] += 1
            continue
        if cve_id in seen_ids:
            stats["duplicate_id"] += 1
            continue

        metric_entries = cve.get("metrics", {}).get("cvssMetricV40", [])
        if not metric_entries:
            stats["without_cvss4"] += 1
            continue
        if len(metric_entries) != 1:
            # No ocurre en el feed observado, pero evitamos elegir una etiqueta
            # de forma silenciosa si el formato cambia en el futuro.
            stats["ambiguous_cvss4"] += 1
            continue

        metric = metric_entries[0]
        cvss_data = metric.get("cvssData", {})
        vector = cvss_data.get("vectorString")
        if not isinstance(vector, str):
            stats["missing_vector"] += 1
            continue
        parsed = parse_base_vector(vector)
        if parsed is None:
            stats["invalid_vector"] += 1
            continue

        description = english_description(cve)
        if description is None:
            stats["missing_description"] += 1
            continue

        labels = tuple(VALUE_TO_ID[name][parsed[name]] for name in METRICS)
        records.append(
            CvssRecord(
                cve_id=cve_id,
                description=description,
                labels=labels,
                vector=vector,
                source_type=str(metric.get("type", "Unknown")),
            )
        )
        seen_ids.add(cve_id)
        stats["usable"] += 1
        stats[f"source_{metric.get('type', 'Unknown')}"] += 1

    return records, dict(stats)


def label_distribution(records: Iterable[CvssRecord]) -> dict[str, Counter[str]]:
    distribution = {metric: Counter() for metric in METRICS}
    for record in records:
        for index, metric in enumerate(METRICS):
            value = METRIC_VALUES[metric][record.labels[index]]
            distribution[metric][value] += 1
    return distribution


def print_dataset_report(records: Sequence[CvssRecord], stats: dict[str, int]) -> None:
    print("\nDataset NVD CVSS 4.0")
    print("=" * 72)
    print(f"CVE totales:                 {stats.get('total_cves', 0):>8}")
    print(f"CVE utilizables con CVSS 4: {len(records):>8}")
    print(f"Sin CVSS 4.0:               {stats.get('without_cvss4', 0):>8}")
    print(f"Etiquetas ambiguas:         {stats.get('ambiguous_cvss4', 0):>8}")
    print(f"Evaluaciones Primary:       {stats.get('source_Primary', 0):>8}")
    print(f"Evaluaciones Secondary:     {stats.get('source_Secondary', 0):>8}")
    print("\nDistribución por métrica")
    print("-" * 72)
    for metric, counts in label_distribution(records).items():
        formatted = "  ".join(f"{value}:{counts[value]}" for value in METRIC_VALUES[metric])
        print(f"{metric:>2}  {formatted}")


def split_records(
    records: Sequence[CvssRecord], seed: int, limit: int | None
) -> tuple[list[CvssRecord], list[CvssRecord], list[CvssRecord]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    if limit is not None:
        shuffled = shuffled[: min(limit, len(shuffled))]
    if len(shuffled) < 100:
        raise ValueError("Se necesitan al menos 100 registros para la demostración.")

    train_end = max(1, int(len(shuffled) * 0.80))
    validation_end = max(train_end + 1, int(len(shuffled) * 0.90))
    return (
        shuffled[:train_end],
        shuffled[train_end:validation_end],
        shuffled[validation_end:],
    )


def require_ml() -> tuple[Any, Any, Any]:
    try:
        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Faltan dependencias. Instálalas con:\n\n"
            "  python3 -m pip install 'torch>=2.1' 'transformers>=4.48' safetensors\n"
        ) from exc
    print(
        f"PyTorch {torch.__version__} · Transformers {transformers.__version__}",
        file=sys.stderr,
    )
    return torch, AutoModel, AutoTokenizer


def choose_device(torch: Any, requested: str) -> Any:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model_class(torch: Any, auto_model: Any) -> type:
    class SecureBertCvss4(torch.nn.Module):
        def __init__(self, model_id: str, dropout: float = 0.15) -> None:
            super().__init__()
            self.model_id = model_id
            self.encoder = auto_model.from_pretrained(model_id)
            hidden_size = int(self.encoder.config.hidden_size)
            self.dropout = torch.nn.Dropout(dropout)
            self.heads = torch.nn.ModuleDict(
                {
                    metric: torch.nn.Linear(hidden_size, len(values))
                    for metric, values in METRIC_VALUES.items()
                }
            )

        def forward(self, input_ids: Any, attention_mask: Any) -> dict[str, Any]:
            output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden = output.last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled = self.dropout(pooled)
            return {metric: head(pooled) for metric, head in self.heads.items()}

    return SecureBertCvss4


def build_dataset_class(torch: Any) -> type:
    class EncodedCvssDataset(torch.utils.data.Dataset):
        def __init__(self, records: Sequence[CvssRecord], tokenizer: Any, max_length: int) -> None:
            self.records = list(records)
            encoded = tokenizer(
                [record.description for record in self.records],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            self.input_ids = encoded["input_ids"]
            self.attention_mask = encoded["attention_mask"]
            self.labels = torch.tensor([record.labels for record in self.records], dtype=torch.long)

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return {
                "input_ids": self.input_ids[index],
                "attention_mask": self.attention_mask[index],
                "labels": self.labels[index],
            }

    return EncodedCvssDataset


def class_weights(torch: Any, records: Sequence[CvssRecord], device: Any) -> dict[str, Any]:
    weights: dict[str, Any] = {}
    for metric_index, metric in enumerate(METRICS):
        counts = [0] * len(METRIC_VALUES[metric])
        for record in records:
            counts[record.labels[metric_index]] += 1
        total = sum(counts)
        raw = [total / (len(counts) * max(count, 1)) for count in counts]
        # Evita que una clase extremadamente rara desestabilice la demo corta.
        clipped = [min(value, 8.0) for value in raw]
        weights[metric] = torch.tensor(clipped, dtype=torch.float32, device=device)
    return weights


def loss_for_batch(
    torch: Any,
    logits: dict[str, Any],
    labels: Any,
    weights: dict[str, Any],
) -> Any:
    losses = []
    for metric_index, metric in enumerate(METRICS):
        losses.append(
            torch.nn.functional.cross_entropy(
                logits[metric], labels[:, metric_index], weight=weights[metric]
            )
        )
    return torch.stack(losses).mean()


def collect_logits(
    torch: Any, model: Any, loader: Any, device: Any
) -> tuple[dict[str, Any], Any, float]:
    model.eval()
    outputs = {metric: [] for metric in METRICS}
    targets = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            for metric in METRICS:
                outputs[metric].append(logits[metric].detach().cpu())
            targets.append(batch["labels"].cpu())
    elapsed = time.perf_counter() - started
    return (
        {metric: torch.cat(chunks, dim=0) for metric, chunks in outputs.items()},
        torch.cat(targets, dim=0),
        elapsed,
    )


def calibrate_temperatures(torch: Any, logits: dict[str, Any], labels: Any) -> dict[str, float]:
    temperatures: dict[str, float] = {}
    for metric_index, metric in enumerate(METRICS):
        metric_logits = logits[metric].float()
        metric_labels = labels[:, metric_index]
        log_temperature = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

        def closure() -> Any:
            optimizer.zero_grad()
            temperature = log_temperature.exp().clamp(0.05, 10.0)
            loss = torch.nn.functional.cross_entropy(metric_logits / temperature, metric_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        temperatures[metric] = float(log_temperature.detach().exp().clamp(0.05, 10.0).item())
    return temperatures


def macro_f1(torch: Any, predictions: Any, labels: Any, classes: int) -> float:
    scores = []
    for class_id in range(classes):
        predicted = predictions == class_id
        actual = labels == class_id
        true_positive = int((predicted & actual).sum().item())
        false_positive = int((predicted & ~actual).sum().item())
        false_negative = int((~predicted & actual).sum().item())
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append((2 * true_positive / denominator) if denominator else 0.0)
    return sum(scores) / len(scores)


def expected_calibration_error(torch: Any, probabilities: Any, labels: Any, bins: int = 10) -> float:
    confidence, predictions = probabilities.max(dim=-1)
    correctness = predictions.eq(labels).float()
    result = torch.tensor(0.0)
    boundaries = torch.linspace(0.0, 1.0, bins + 1)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            weight = selected.float().mean()
            gap = (correctness[selected].mean() - confidence[selected].mean()).abs()
            result += weight * gap
    return float(result.item())


def evaluation_report(
    torch: Any,
    logits: dict[str, Any],
    labels: Any,
    temperatures: dict[str, float],
    title: str,
) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    print(f"\n{title}")
    print("=" * 72)
    print(f"{'Métrica':<8}{'Accuracy':>12}{'Macro-F1':>12}{'ECE':>12}{'Temp.':>12}")
    for metric_index, metric in enumerate(METRICS):
        temperature = temperatures.get(metric, 1.0)
        probabilities = torch.softmax(logits[metric] / temperature, dim=-1)
        predictions = probabilities.argmax(dim=-1)
        metric_labels = labels[:, metric_index]
        accuracy = float(predictions.eq(metric_labels).float().mean().item())
        f1 = macro_f1(torch, predictions, metric_labels, len(METRIC_VALUES[metric]))
        ece = expected_calibration_error(torch, probabilities, metric_labels)
        report[metric] = {
            "accuracy": accuracy,
            "macro_f1": f1,
            "ece": ece,
            "temperature": temperature,
        }
        print(f"{metric:<8}{accuracy:>12.3f}{f1:>12.3f}{ece:>12.3f}{temperature:>12.3f}")
    print("-" * 72)
    print(
        f"Promedio{sum(row['accuracy'] for row in report.values()) / len(report):>12.3f}"
        f"{sum(row['macro_f1'] for row in report.values()) / len(report):>12.3f}"
        f"{sum(row['ece'] for row in report.values()) / len(report):>12.3f}"
    )
    return report


def command_inspect(args: argparse.Namespace) -> int:
    records, stats = load_records(args.dataset)
    print_dataset_report(records, stats)
    return 0


def command_train(args: argparse.Namespace) -> int:
    torch, auto_model, auto_tokenizer = require_ml()
    seed_everything(torch, args.seed)
    device = choose_device(torch, args.device)
    print(f"Dispositivo: {device}")

    records, stats = load_records(args.dataset)
    print_dataset_report(records, stats)

    if args.quick:
        limit = min(args.limit or 2200, 2200)
        epochs = 1
        max_length = min(args.max_length, 160)
        print("\nModo rápido: máximo 2.200 casos, 1 época y 160 tokens.")
    else:
        limit = args.limit
        epochs = args.epochs
        max_length = args.max_length

    train_records, validation_records, test_records = split_records(records, args.seed, limit)
    print(
        f"Split: train={len(train_records)} · validación={len(validation_records)} "
        f"· test={len(test_records)}"
    )

    print(f"\nCargando tokenizer y modelo: {args.model}")
    tokenizer = auto_tokenizer.from_pretrained(args.model)
    model_class = build_model_class(torch, auto_model)
    model = model_class(args.model, dropout=args.dropout)
    if args.linear_probe:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
        print("Modo linear probe: el encoder queda congelado.")
    elif hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
    model.to(device)

    dataset_class = build_dataset_class(torch)
    print("Tokenizando descripciones...")
    train_dataset = dataset_class(train_records, tokenizer, max_length)
    validation_dataset = dataset_class(validation_records, tokenizer, max_length)
    test_dataset = dataset_class(test_records, tokenizer, max_length)

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **loader_options)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, shuffle=False, **loader_options)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, **loader_options)

    weights = class_weights(torch, train_records, device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = max(1, len(train_loader) * epochs)
    warmup_steps = max(1, int(total_steps * 0.10))

    def learning_rate(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate)
    global_step = 0
    print("\nComenzando entrenamiento...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        epoch_started = time.perf_counter()
        for batch_index, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_for_batch(torch, logits, labels, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            running_loss += float(loss.detach().item())
            if batch_index == 1 or batch_index % 20 == 0 or batch_index == len(train_loader):
                print(
                    f"Época {epoch + 1}/{epochs} · batch {batch_index}/{len(train_loader)} "
                    f"· loss={running_loss / batch_index:.4f}"
                )
        print(f"Época terminada en {time.perf_counter() - epoch_started:.1f} s")

    print("\nCalculando calibración en validación...")
    validation_logits, validation_labels, validation_time = collect_logits(
        torch, model, validation_loader, device
    )
    temperatures = calibrate_temperatures(torch, validation_logits, validation_labels)
    evaluation_report(
        torch, validation_logits, validation_labels, temperatures, "Validación calibrada"
    )

    test_logits, test_labels, test_time = collect_logits(torch, model, test_loader, device)
    test_report = evaluation_report(torch, test_logits, test_labels, temperatures, "Test reservado")

    checkpoint = {
        "format_version": 1,
        "model_id": args.model,
        "model_state": model.state_dict(),
        "metric_values": METRIC_VALUES,
        "temperatures": temperatures,
        "max_length": max_length,
        "seed": args.seed,
        "linear_probe": args.linear_probe,
        "records": {
            "train": len(train_records),
            "validation": len(validation_records),
            "test": len(test_records),
        },
        "test_report": test_report,
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.checkpoint)
    print(f"\nCheckpoint guardado: {args.checkpoint}")
    print(
        f"Latencia de evaluación por caso: "
        f"validación={1000 * validation_time / len(validation_records):.1f} ms · "
        f"test={1000 * test_time / len(test_records):.1f} ms"
    )
    print("\nSiguiente paso:")
    print(
        f"  python {Path(__file__).name} predict --checkpoint {args.checkpoint.name} "
        "--text \"Remote attackers can execute arbitrary code...\""
    )
    return 0


def load_api_request(path: Path) -> dict[str, Any]:
    """Carga el contrato HTTP de la demo sin agregar una dependencia en Pydantic."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SystemExit(f"Request JSON no encontrado: {path}") from error
    except json.JSONDecodeError as error:
        raise SystemExit(f"Request JSON inválido en {path}: {error}") from error

    if not isinstance(payload, dict):
        raise SystemExit("El request JSON debe ser un objeto.")

    limits = {
        "vulnerability_description": 8_000,
        "application_name": 200,
        "application_context": 12_000,
    }
    for field, maximum in limits.items():
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise SystemExit(f"El campo `{field}` es obligatorio y debe ser texto no vacío.")
        if len(value) > maximum:
            raise SystemExit(
                f"El campo `{field}` supera el máximo de {maximum} caracteres."
            )

    vulnerability_id = payload.get("vulnerability_id")
    if vulnerability_id is not None:
        if not isinstance(vulnerability_id, str) or not vulnerability_id.strip():
            raise SystemExit("`vulnerability_id` debe ser texto no vacío o null.")
        if len(vulnerability_id) > 100:
            raise SystemExit("`vulnerability_id` supera el máximo de 100 caracteres.")

    return payload


def command_predict(args: argparse.Namespace) -> int:
    torch, auto_model, auto_tokenizer = require_ml()
    device = choose_device(torch, args.device)
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint no encontrado: {args.checkpoint}\nEjecuta primero `train`.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_id = checkpoint["model_id"]
    tokenizer = auto_tokenizer.from_pretrained(model_id)
    model_class = build_model_class(torch, auto_model)
    model = model_class(model_id)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    request_metadata: dict[str, Any] | None = None
    description = args.text
    if args.request_file is not None:
        payload = load_api_request(args.request_file)
        description = payload["vulnerability_description"].strip()
        application_context = payload["application_context"].strip()
        description = f"{description}\n\nApplication context: {application_context}"
        request_metadata = {
            "vulnerability_id": payload.get("vulnerability_id"),
            "application_name": payload["application_name"],
            "source": str(args.request_file),
        }
    elif args.text_file is not None:
        description = args.text_file.read_text(encoding="utf-8").strip()
    if not description:
        raise SystemExit("Indica `--text`, `--text-file` o `--request-file`.")
    if args.context and args.request_file is None:
        description = f"{description}\n\nApplication context: {args.context}"

    encoded = tokenizer(
        description,
        truncation=True,
        max_length=int(checkpoint["max_length"]),
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
    elapsed_ms = (time.perf_counter() - started) * 1000

    metrics: dict[str, dict[str, Any]] = {}
    low_confidence: list[str] = []
    vector_parts = ["CVSS:4.0"]
    for metric in METRICS:
        temperature = float(checkpoint["temperatures"].get(metric, 1.0))
        probabilities = torch.softmax(logits[metric].cpu() / temperature, dim=-1)[0]
        class_id = int(probabilities.argmax().item())
        value = METRIC_VALUES[metric][class_id]
        confidence = float(probabilities[class_id].item())
        metrics[metric] = {"value": value, "confidence": round(confidence, 4)}
        vector_parts.append(f"{metric}:{value}")
        if confidence < args.confidence_threshold:
            low_confidence.append(metric)

    result = {
        "engine": "securebert-cvss4-demo",
        "model": model_id,
        "request": request_metadata,
        "vector": "/".join(vector_parts),
        "metrics": metrics,
        "illustrative_gate": {
            "threshold": args.confidence_threshold,
            "decision": "ACCEPT_ENCODER" if not low_confidence else "FALLBACK_LLM",
            "low_confidence_metrics": low_confidence,
            "warning": "Umbral educativo; todavía no es una política de producción.",
        },
        "latency_ms": round(elapsed_ms, 2),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="SecureBERT 2.0 multi-head para una demo CVSS 4.0.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    commands = root.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect", help="Inspecciona el dataset sin instalar ML.")
    inspect.add_argument("--dataset", type=Path, default=default_dataset_path())
    inspect.set_defaults(handler=command_inspect)

    train = commands.add_parser("train", help="Entrena y calibra el clasificador multi-head.")
    train.add_argument("--dataset", type=Path, default=default_dataset_path())
    train.add_argument("--checkpoint", type=Path, default=default_checkpoint_path())
    train.add_argument("--model", default=MODEL_ID)
    train.add_argument("--device", default="auto", help="auto, mps, cuda o cpu")
    train.add_argument("--quick", action="store_true", help="Prueba corta para entender el flujo.")
    train.add_argument("--linear-probe", action="store_true", help="Entrena solo las once cabezas.")
    train.add_argument("--limit", type=int, default=None, help="Límite total de registros; 0 no limita.")
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--max-length", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=2e-5)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--dropout", type=float, default=0.15)
    train.add_argument("--seed", type=int, default=42)
    train.set_defaults(handler=command_train)

    predict = commands.add_parser("predict", help="Predice once métricas con un checkpoint.")
    predict.add_argument("--checkpoint", type=Path, default=default_checkpoint_path())
    predict.add_argument("--device", default="auto", help="auto, mps, cuda o cpu")
    input_group = predict.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text")
    input_group.add_argument("--text-file", type=Path)
    input_group.add_argument(
        "--request-file",
        type=Path,
        help="JSON con el mismo contrato de entrada que POST /v1/assessments.",
    )
    predict.add_argument("--context", default="")
    predict.add_argument("--confidence-threshold", type=float, default=0.70)
    predict.set_defaults(handler=command_predict)

    return root


def main() -> int:
    args = parser().parse_args()
    if getattr(args, "limit", None) == 0:
        args.limit = None
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.", file=sys.stderr)
        return 130
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
