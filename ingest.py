"""
Pipeline de Ingesta a Pinecone.

Lee documentos técnicos de /data, los fragmenta con
RecursiveCharacterTextSplitter (500-800 tokens, punto medio recomendado
por la consigna) y los sube a un índice Serverless de Pinecone,
incluyendo metadata avanzada: fuente, categoría, número de chunk, y el
texto original del fragmento (para no depender de una base de datos
relacional adicional al recuperar resultados).
"""

import logging
import os
from pathlib import Path
from typing import List

import tiktoken
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embeddings import get_embeddings
from setup_pinecone import INDEX_NAME, asegurar_indice, get_pinecone_client

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# El namespace separa lógicamente los datos dentro de un mismo índice.
# La consigna advierte explícitamente: ignorar el namespace en
# aplicaciones con distintos tipos de datos hace que la búsqueda sea
# ruidosa y lenta (mezclás todo en un solo espacio de búsqueda).
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "documentos-tecnicos")

# Chunking: entre 500 y 800 tokens (punto medio recomendado por la
# consigna: chunks muy chicos pierden contexto semántico, muy grandes
# diluyen la precisión del embedding).
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "650"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "100"))

BATCH_SIZE = 100  # tamaño de lote para el upsert a Pinecone

_encoding = tiktoken.get_encoding("cl100k_base")


def _contar_tokens(texto: str) -> int:
    """Cuenta tokens reales (no caracteres ni palabras) usando tiktoken."""
    return len(_encoding.encode(texto))


# Categoría asignada a cada archivo fuente, para poblar metadata avanzada
# (además de fuente y número de página/chunk).
_CATEGORIAS_POR_ARCHIVO = {
    "01_arquitectura.md": "arquitectura",
    "02_seguridad.md": "seguridad",
    "03_testing.md": "testing",
    "04_deployment.md": "deployment",
}


# ---------------------------------------------------------------------------
# Carga y chunking (misma lógica que el RAG local, adaptada con categoría)
# ---------------------------------------------------------------------------

def cargar_documentos(data_dir: Path = DATA_DIR) -> List[Document]:
    """Lee todos los .txt/.md de la carpeta indicada, con metadata de categoría."""
    if not data_dir.exists():
        raise FileNotFoundError(f"No se encontró la carpeta de datos: {data_dir}")

    archivos = sorted([*data_dir.glob("*.txt"), *data_dir.glob("*.md")])
    if not archivos:
        raise FileNotFoundError(f"No se encontraron archivos .txt/.md en {data_dir}")

    documentos: List[Document] = []
    for archivo in archivos:
        contenido = archivo.read_text(encoding="utf-8")
        categoria = _CATEGORIAS_POR_ARCHIVO.get(archivo.name, "general")
        doc = Document(
            page_content=contenido,
            metadata={"source": archivo.name, "categoria": categoria},
        )
        documentos.append(doc)
        logger.info(f"Cargado: {archivo.name} (categoría: {categoria})")

    logger.info(f"Total de documentos cargados: {len(documentos)}")
    return documentos


def fragmentar_documentos(documentos: List[Document]) -> List[Document]:
    """Fragmenta los documentos y numera cada chunk dentro de su archivo de origen."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        length_function=_contar_tokens,
    )
    chunks = splitter.split_documents(documentos)

    # Numeramos cada chunk dentro de su archivo de origen (equivalente al
    # "número de página" que pide la consigna para PDFs) — permite armar
    # IDs únicos y determinísticos para el upsert en Pinecone.
    contador_por_fuente: dict = {}
    for chunk in chunks:
        fuente = chunk.metadata["source"]
        indice = contador_por_fuente.get(fuente, 0)
        chunk.metadata["chunk_index"] = indice
        contador_por_fuente[fuente] = indice + 1

    logger.info(
        f"Documentos fragmentados en {len(chunks)} chunks "
        f"(chunk_size={CHUNK_SIZE_TOKENS} tokens, overlap={CHUNK_OVERLAP_TOKENS} tokens)"
    )
    return chunks


# ---------------------------------------------------------------------------
# Persistencia en Pinecone
# ---------------------------------------------------------------------------

def _namespace_ya_poblado(pc, namespace: str) -> bool:
    """Verifica si el namespace ya tiene vectores, para no reindexar de más."""
    index = pc.Index(INDEX_NAME)
    stats = index.describe_index_stats()
    stats_namespace = stats.get("namespaces", {}).get(namespace)
    return bool(stats_namespace and stats_namespace.get("vector_count", 0) > 0)


def ingestar(forzar_reindexado: bool = False) -> None:
    """
    Punto de entrada principal del pipeline de ingesta.

    Verifica que el índice exista (lo crea si hace falta), chequea si el
    namespace ya tiene datos (evita reindexar de más), y si no, carga,
    fragmenta, embebe y sube los documentos en lotes.

    Args:
        forzar_reindexado: si es True, reindexa aunque el namespace ya
                            tenga vectores cargados
    """
    pc = get_pinecone_client()
    asegurar_indice(pc)

    if _namespace_ya_poblado(pc, NAMESPACE) and not forzar_reindexado:
        logger.info(
            f"El namespace '{NAMESPACE}' del índice '{INDEX_NAME}' ya tiene "
            f"vectores cargados. No se reindexa (usá forzar_reindexado=True "
            f"si necesitás hacerlo de nuevo)."
        )
        return

    documentos = cargar_documentos()
    chunks = fragmentar_documentos(documentos)
    embeddings_model = get_embeddings()
    index = pc.Index(INDEX_NAME)

    logger.info(f"Generando embeddings para {len(chunks)} chunks...")
    vectores = embeddings_model.embed_documents([c.page_content for c in chunks])

    registros = []
    for chunk, vector in zip(chunks, vectores):
        registros.append(
            {
                "id": f"{chunk.metadata['source']}-{chunk.metadata['chunk_index']}",
                "values": vector,
                "metadata": {
                    # Guardamos el texto original en la metadata: así, al
                    # recuperar resultados, no hace falta ninguna consulta
                    # adicional a una base de datos relacional para
                    # obtener el contenido del fragmento.
                    "text": chunk.page_content,
                    "source": chunk.metadata["source"],
                    "categoria": chunk.metadata["categoria"],
                    "chunk_index": chunk.metadata["chunk_index"],
                },
            }
        )

    logger.info(
        f"Subiendo {len(registros)} vectores al namespace '{NAMESPACE}' "
        f"en lotes de {BATCH_SIZE}..."
    )
    for inicio in range(0, len(registros), BATCH_SIZE):
        lote = registros[inicio : inicio + BATCH_SIZE]
        index.upsert(vectors=lote, namespace=NAMESPACE)

    logger.info(
        f"Ingesta completa: {len(registros)} vectores persistidos en "
        f"'{INDEX_NAME}' / namespace '{NAMESPACE}'"
    )


if __name__ == "__main__":
    ingestar()
