"""
Setup de Infraestructura: verifica si el índice de Pinecone ya existe y,
si no, lo crea en modo Serverless con la dimensión correcta para el
proveedor de embeddings configurado.
"""

import logging
import os
import time

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from embeddings import EMBEDDING_DIMENSION

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

INDEX_NAME = os.getenv("INDEX_NAME", "entregable-4-rag")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")


def get_pinecone_client() -> Pinecone:
    """Crea el cliente de Pinecone a partir de la API key del .env."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError(
            "Falta PINECONE_API_KEY en el .env. Conseguila en "
            "https://app.pinecone.io/ (tienen un plan gratis Serverless)."
        )
    return Pinecone(api_key=api_key)


def asegurar_indice(pc: Pinecone = None, forzar_recreacion: bool = False) -> None:
    """
    Verifica si el índice ya existe antes de crearlo (no lo recrea en
    cada corrida). Si existe pero con una dimensión distinta a la que
    necesita el proveedor de embeddings configurado, lanza un error claro
    en vez de dejar que Pinecone rechace los upserts más adelante con un
    mensaje críptico — este es justo el "mismatch de dimensiones" que
    advierte la consigna como error común.

    Args:
        pc: cliente de Pinecone ya creado (opcional, se crea uno si no se pasa)
        forzar_recreacion: si es True, borra y recrea el índice desde cero

    Raises:
        ValueError: si el índice existe con una dimensión incompatible
    """
    pc = pc or get_pinecone_client()
    indices_existentes = [idx["name"] for idx in pc.list_indexes()]

    if INDEX_NAME in indices_existentes:
        info = pc.describe_index(INDEX_NAME)

        if info.dimension != EMBEDDING_DIMENSION:
            if not forzar_recreacion:
                raise ValueError(
                    f"El índice '{INDEX_NAME}' ya existe con dimensión "
                    f"{info.dimension}, pero EMBEDDINGS_PROVIDER="
                    f"'{os.getenv('EMBEDDINGS_PROVIDER', 'local')}' necesita "
                    f"dimensión {EMBEDDING_DIMENSION}. Esto es exactamente "
                    f"el 'mismatch de dimensiones' que advierte la consigna: "
                    f"cambiá INDEX_NAME en tu .env para usar un índice nuevo, "
                    f"o llamá a asegurar_indice(forzar_recreacion=True) si "
                    f"estás seguro de que querés borrar el índice actual."
                )
            logger.warning(
                f"Borrando índice '{INDEX_NAME}' (dimensión incompatible) "
                f"para recrearlo con dimensión {EMBEDDING_DIMENSION}..."
            )
            pc.delete_index(INDEX_NAME)
        else:
            logger.info(
                f"El índice '{INDEX_NAME}' ya existe (dimensión "
                f"{info.dimension}, coincide con la configurada). No se recrea."
            )
            return

    logger.info(
        f"Creando índice Serverless '{INDEX_NAME}' "
        f"(dimensión={EMBEDDING_DIMENSION}, cloud={PINECONE_CLOUD}, "
        f"region={PINECONE_REGION})..."
    )
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )

    # Espera activa a que el índice quede disponible antes de devolver el control
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        logger.info("Esperando a que el índice quede listo...")
        time.sleep(2)

    logger.info(f"Índice '{INDEX_NAME}' creado y listo para usarse.")


if __name__ == "__main__":
    asegurar_indice()
