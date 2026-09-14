"""
Módulo centralizado para el modelo de embeddings.

Igual que en el módulo de RAG local: existe para garantizar que el
pipeline de ingesta y el recuperador usen exactamente el mismo modelo de
embeddings. Acá además cumple un rol extra: expone EMBEDDING_DIMENSION,
que setup_pinecone.py usa para crear el índice con la dimensión correcta.

Esto es clave porque el error más común al trabajar con Pinecone es el
"mismatch de dimensiones": crear un índice con dimensión 1536 (típica de
OpenAI text-embedding-3-small) e intentar subir vectores de 384
dimensiones (típica de modelos locales chicos), o viceversa. Pinecone
rechaza el upsert en ese caso. Centralizar el proveedor Y su dimensión en
un solo lugar hace que ese error sea imposible por diseño.
"""

import os
from typing import List

from langchain_core.embeddings import Embeddings

# "local" (default, gratis, sin API key, fastembed) u "openai" (requiere
# OPENAI_API_KEY con crédito cargado, mejor calidad en inglés/multilingüe)
EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "local")

LOCAL_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Dimensión de salida de cada proveedor. setup_pinecone.py crea el índice
# con este valor exacto, y lo vuelve a chequear en cada corrida para
# detectar si alguien cambió de proveedor sin recrear el índice.
_DIMENSIONES_POR_PROVEEDOR = {
    "local": 384,    # BAAI/bge-small-en-v1.5
    "openai": 1536,  # text-embedding-3-small
}

if EMBEDDINGS_PROVIDER not in _DIMENSIONES_POR_PROVEEDOR:
    raise ValueError(
        f"EMBEDDINGS_PROVIDER no soportado: '{EMBEDDINGS_PROVIDER}'. "
        f"Usá 'local' o 'openai'."
    )

EMBEDDING_DIMENSION = _DIMENSIONES_POR_PROVEEDOR[EMBEDDINGS_PROVIDER]


class FastEmbedEmbeddings(Embeddings):
    """
    Wrapper mínimo que adapta la librería fastembed (embeddings locales
    vía ONNX Runtime, sin depender de PyTorch) a la interfaz Embeddings
    de LangChain.
    """

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        return next(iter(self._model.embed([text]))).tolist()


def get_embeddings():
    """
    Devuelve siempre la misma instancia de modelo de embeddings, según
    EMBEDDINGS_PROVIDER.

    Los imports de cada proveedor son diferidos (dentro de la función):
    así, usando "local" nunca hace falta tener instalado ni configurado
    nada de OpenAI, y viceversa.
    """
    if EMBEDDINGS_PROVIDER == "local":
        return FastEmbedEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)

    elif EMBEDDINGS_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
