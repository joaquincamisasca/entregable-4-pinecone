"""
RAGSystem: Recuperador Híbrido que combina búsqueda vectorial (semántica,
vía Pinecone) con búsqueda léxica (BM25), usando un EnsembleRetriever de
LangChain.

La búsqueda vectorial es fuerte para similitud de significado, pero
puede fallar con términos técnicos exactos, nombres propios o
identificadores específicos (ej: "HashiCorp Vault", "OAuth2", "Kafka")
porque los diluye en el espacio semántico. BM25 (búsqueda léxica clásica,
basada en coincidencia de términos) complementa exactamente ese punto
débil. Combinar ambos mejora la precisión en ese tipo de consultas.
"""

import logging
import os
import re
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from embeddings import get_embeddings
from ingest import NAMESPACE, cargar_documentos, fragmentar_documentos
from setup_pinecone import INDEX_NAME, get_pinecone_client

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TOP_K = int(os.getenv("RAG_TOP_K", "5"))


def _tokenizar_para_bm25(texto: str) -> List[str]:
    """
    Tokenizador para BM25: extrae solo palabras (letras/números/tildes),
    en minúsculas.

    El preprocesador por default de BM25Retriever solo hace texto.split()
    por espacios, sin sacar puntuación — así que términos con formato
    Markdown pegado (ej: "**HashiCorp**" o "Vault**,") nunca calzan con la
    consulta limpia del usuario ("HashiCorp Vault"). Este tokenizador
    evita ese problema extrayendo palabras "limpias" con una regex.
    """
    return re.findall(r"\w+", texto.lower(), flags=re.UNICODE)


class PineconeRetriever(BaseRetriever):
    """
    Wrapper mínimo que adapta el índice de Pinecone (consultado con el
    SDK nativo) a la interfaz BaseRetriever de LangChain, para poder
    combinarlo con BM25Retriever dentro de un EnsembleRetriever.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: object
    embeddings: object
    namespace: str
    top_k: int = 5

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        vector_consulta = self.embeddings.embed_query(query)
        resultados = self.index.query(
            vector=vector_consulta,
            top_k=self.top_k,
            namespace=self.namespace,
            include_metadata=True,
        )

        documentos = []
        for match in resultados.get("matches", []):
            metadata = dict(match.get("metadata", {}))
            texto = metadata.pop("text", "")
            metadata["score_vectorial"] = match.get("score")
            documentos.append(Document(page_content=texto, metadata=metadata))

        return documentos


class RAGSystem:
    """
    Encapsula el recuperador híbrido completo: búsqueda vectorial en
    Pinecone + búsqueda léxica BM25, combinadas mediante un
    EnsembleRetriever.

    El corpus de BM25 se construye a partir de los mismos documentos
    locales que se subieron a Pinecone durante la ingesta (misma
    fragmentación, mismo contenido), garantizando que ambos recuperadores
    buscan sobre exactamente el mismo material.
    """

    def __init__(self, top_k: int = TOP_K, pesos: Tuple[float, float] = (0.5, 0.5)):
        """
        Args:
            top_k: cantidad de documentos a devolver por buscar()
            pesos: peso relativo (vectorial, léxico) dentro del
                   EnsembleRetriever. (0.5, 0.5) por default pondera
                   ambos por igual.
        """
        self.top_k = top_k

        pc = get_pinecone_client()
        index = pc.Index(INDEX_NAME)
        embeddings_model = get_embeddings()

        vector_retriever = PineconeRetriever(
            index=index,
            embeddings=embeddings_model,
            namespace=NAMESPACE,
            top_k=top_k,
        )

        documentos = cargar_documentos()
        chunks = fragmentar_documentos(documentos)
        bm25_retriever = BM25Retriever.from_documents(
            chunks, preprocess_func=_tokenizar_para_bm25
        )
        bm25_retriever.k = top_k

        self.retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=list(pesos),
        )

        logger.info(
            f"RAGSystem inicializado (top_k={top_k}, pesos vectorial/léxico={pesos})"
        )

    def buscar(self, query: str) -> List[Document]:
        """
        Ejecuta la búsqueda híbrida y devuelve los top-k documentos,
        combinando resultados léxicos (BM25) y semánticos (Pinecone).

        Args:
            query: consulta del usuario en lenguaje natural

        Returns:
            Lista de hasta top_k Document, ya combinados y reordenados
            por el EnsembleRetriever (algoritmo de fusión por rango)

        Raises:
            ValueError: si la consulta está vacía
        """
        if not query or not query.strip():
            raise ValueError("La consulta no puede estar vacía.")

        resultados = self.retriever.invoke(query)
        return resultados[: self.top_k]
