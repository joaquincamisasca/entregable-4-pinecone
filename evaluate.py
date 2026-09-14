"""
Script de Evaluación: calcula Precision@5 y Recall@5 del recuperador
híbrido (RAGSystem) contra un pequeño "Golden Set" de preguntas con
documento fuente esperado conocido de antemano.

Definiciones usadas acá (a nivel de documento fuente, no de chunk
individual, ya que cada pregunta del golden set tiene un único documento
relevante conocido):

- Recall@5 (por pregunta): 1 si el documento esperado aparece entre los
  5 fragmentos recuperados, 0 si no. El Recall@5 global es el promedio
  sobre todas las preguntas del golden set.

- Precision@5 (por pregunta): de los 5 fragmentos recuperados, qué
  proporción pertenece al documento esperado (fragmentos "útiles"). El
  Precision@5 global es el promedio sobre todas las preguntas.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

from rag_system import RAGSystem

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"


def cargar_golden_set(path: Path = GOLDEN_SET_PATH) -> List[Dict]:
    """Carga el golden set de preguntas + documento esperado desde un JSON."""
    with open(path, encoding="utf-8") as f:
        golden_set = json.load(f)

    if not golden_set:
        raise ValueError(f"El golden set en {path} está vacío.")

    return golden_set


def evaluar(rag: RAGSystem, golden_set: List[Dict]) -> Tuple[List[Dict], float, float]:
    """
    Corre cada pregunta del golden set contra el RAGSystem y calcula
    Precision@5 y Recall@5.

    Args:
        rag: instancia ya inicializada de RAGSystem
        golden_set: lista de {"pregunta": ..., "documento_id_esperado": ...}

    Returns:
        (resultados_detallados, recall_at_5_global, precision_at_5_promedio)
    """
    resultados = []
    aciertos_recall = 0
    precisiones = []

    for caso in golden_set:
        pregunta = caso["pregunta"]
        esperado = caso["documento_id_esperado"]

        docs_recuperados = rag.buscar(pregunta)
        fuentes_recuperadas = [d.metadata.get("source") for d in docs_recuperados]

        acierto_recall = esperado in fuentes_recuperadas
        relevantes_en_top5 = sum(1 for f in fuentes_recuperadas if f == esperado)
        precision_at_5 = (
            relevantes_en_top5 / len(fuentes_recuperadas) if fuentes_recuperadas else 0.0
        )

        if acierto_recall:
            aciertos_recall += 1
        precisiones.append(precision_at_5)

        resultados.append(
            {
                "pregunta": pregunta,
                "esperado": esperado,
                "recuperados": fuentes_recuperadas,
                "acierto_recall": acierto_recall,
                "precision_at_5": precision_at_5,
            }
        )

        logger.info(
            f"'{pregunta}' -> esperado={esperado}, "
            f"recuperados={fuentes_recuperadas}, acierto={acierto_recall}"
        )

    recall_at_5 = aciertos_recall / len(golden_set) if golden_set else 0.0
    precision_at_5_promedio = sum(precisiones) / len(precisiones) if precisiones else 0.0

    return resultados, recall_at_5, precision_at_5_promedio


def imprimir_reporte(
    resultados: List[Dict], recall_at_5: float, precision_promedio: float
) -> None:
    """Imprime en consola un resumen legible de los resultados de evaluación."""
    print(f"\n{'='*70}")
    print("Reporte de Evaluación — Recuperador Híbrido (BM25 + Pinecone)")
    print(f"{'='*70}\n")

    for r in resultados:
        estado = "✓" if r["acierto_recall"] else "✗"
        print(f"{estado} {r['pregunta']}")
        print(f"    Documento esperado:    {r['esperado']}")
        print(f"    Documentos recuperados: {r['recuperados']}")
        print(f"    Precision@5:            {r['precision_at_5']:.2f}\n")

    aciertos = sum(1 for r in resultados if r["acierto_recall"])
    print(f"{'-'*70}")
    print(f"Recall@5 global:      {recall_at_5:.2%}  ({aciertos}/{len(resultados)} preguntas)")
    print(f"Precision@5 promedio: {precision_promedio:.2%}")
    print(f"{'-'*70}\n")


if __name__ == "__main__":
    golden_set = cargar_golden_set()
    rag = RAGSystem()
    resultados, recall_at_5, precision_promedio = evaluar(rag, golden_set)
    imprimir_reporte(resultados, recall_at_5, precision_promedio)
