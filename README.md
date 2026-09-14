# Sistema RAG Escalable en la Nube (Pinecone + Recuperador Híbrido)

Módulo de Recuperación Escalable: pipeline de ingesta a **Pinecone
Serverless** con metadata avanzada, un **recuperador híbrido** que
combina búsqueda vectorial (semántica) con BM25 (léxica), y un script de
evaluación con **Precision@5** y **Recall@5** contra un Golden Set.

**Módulo 4, Pre-entrega 4** - Programa de AI Engineering @ CodeHouse

---

## 🎯 Por qué un recuperador híbrido

La búsqueda vectorial pura es excelente para similitud de significado,
pero puede fallar con términos técnicos exactos o nombres propios (ej:
"HashiCorp Vault", "OAuth2", "Kafka") porque los diluye en el espacio
semántico. BM25 (búsqueda léxica clásica) complementa exactamente ese
punto débil. Este proyecto combina ambos con un `EnsembleRetriever` de
LangChain.

---

## 📋 Estructura del Proyecto

```
entregable_4_pinecone/
├── data/                     # Dataset de ejemplo (documentación técnica interna)
│   ├── 01_arquitectura.md
│   ├── 02_seguridad.md
│   ├── 03_testing.md
│   └── 04_deployment.md
├── embeddings.py             # Modelo de embeddings + su dimensión (local o OpenAI)
├── setup_pinecone.py          # Verifica/crea el índice Serverless
├── ingest.py                    # Chunking + metadata avanzada + upsert a Pinecone
├── rag_system.py                  # RAGSystem: recuperador híbrido (BM25 + Pinecone)
├── evaluate.py                      # Precision@5 y Recall@5 contra el golden set
├── golden_set.json                    # 5 preguntas de prueba con documento esperado
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Instalación

### 1. Dependencias

```bash
cd entregable_4_pinecone
python -m venv venv
venv\Scripts\activate      # Windows. En Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

### 2. Crear una cuenta y un índice en Pinecone

1. Registrate gratis en [app.pinecone.io](https://app.pinecone.io/) (el
   plan Serverless gratuito alcanza de sobra para este entregable).
2. Andá a **API Keys** en el dashboard y copiá tu key.
3. **No hace falta crear el índice a mano** — `setup_pinecone.py` lo
   crea automáticamente la primera vez que corrés la ingesta, con la
   dimensión correcta según el proveedor de embeddings configurado.

### 3. Configurar variables de entorno

```bash
cp .env.example .env
```

Completá al menos `PINECONE_API_KEY`. Por defecto:
- Los **embeddings** corren local (gratis, sin key, dimensión 384)
- El **índice** se llama `entregable-4-rag`, región `us-east-1` de AWS

Si preferís usar embeddings de OpenAI (dimensión 1536), cambiá
`EMBEDDINGS_PROVIDER=openai` y completá `OPENAI_API_KEY`.

---

## 🏗️ Cómo replicar el índice (paso a paso)

### 1. Crear el índice y cargar los documentos

```bash
python ingest.py
```

Esto hace, en orden:
1. Se conecta a Pinecone con tu API key.
2. Verifica si el índice `INDEX_NAME` ya existe. Si no, lo crea en modo
   **Serverless**, con la dimensión que corresponde al proveedor de
   embeddings configurado (384 para local, 1536 para OpenAI).
3. Verifica si el **namespace** ya tiene vectores cargados (evita
   reindexar de más si ya corriste esto antes).
4. Carga los `.md` de `/data`, los fragmenta (chunks de ~650 tokens,
   punto medio entre 500 y 800 recomendado por la consigna) y genera
   sus embeddings.
5. Sube los vectores a Pinecone en lotes, incluyendo en la metadata:
   `text` (el contenido del chunk, para no depender de una base de
   datos relacional adicional), `source` (archivo de origen),
   `categoria` y `chunk_index`.

Para forzar una re-indexación completa (por ejemplo, si cambiaste el
dataset): `python -c "from ingest import ingestar; ingestar(forzar_reindexado=True)"`.

### 2. Correr una consulta híbrida de prueba

```python
from rag_system import RAGSystem

rag = RAGSystem()
resultados = rag.buscar("¿Cómo se gestionan los secretos de la empresa?")
for doc in resultados:
    print(doc.metadata["source"], "-", doc.page_content[:80])
```

### 3. Evaluar el sistema contra el Golden Set

```bash
python evaluate.py
```

---

## 📊 Golden Set y métricas

`golden_set.json` tiene 5 preguntas, cada una con el documento fuente
donde está la respuesta correcta:

```json
{
  "pregunta": "¿Qué framework se usa para los endpoints REST y por qué se eligió?",
  "documento_id_esperado": "01_arquitectura.md"
}
```

`evaluate.py` corre cada pregunta contra `RAGSystem.buscar()` (que
combina BM25 + Pinecone) y calcula:

- **Recall@5** (por pregunta): 1 si el documento esperado aparece entre
  los 5 fragmentos recuperados, 0 si no. El Recall@5 global es el
  promedio sobre las 5 preguntas.
- **Precision@5** (por pregunta): qué proporción de los 5 fragmentos
  recuperados pertenece al documento esperado. El Precision@5 promedio
  es el promedio sobre las 5 preguntas.

### Resultado de ejemplo (con embeddings locales)

```
======================================================================
Reporte de Evaluación — Recuperador Híbrido (BM25 + Pinecone)
======================================================================

✓ ¿Qué framework se usa para los endpoints REST y por qué se eligió?
    Documento esperado:    01_arquitectura.md
    Documentos recuperados: ['01_arquitectura.md', '01_arquitectura.md', '02_seguridad.md', '03_testing.md', '04_deployment.md']
    Precision@5:            0.40

[...]

----------------------------------------------------------------------
Recall@5 global:      100.00%  (5/5 preguntas)
Precision@5 promedio: 42.00%
----------------------------------------------------------------------
```

*(Los números reales van a depender del proveedor de embeddings y del
contenido exacto de tu índice — corré `python evaluate.py` para ver el
resultado real de tu corrida, y actualizá esta sección con tus números.)*

---

## ⚠️ Errores comunes que este proyecto evita explícitamente

| Error común | Cómo se evita acá |
|---|---|
| Mismatch de dimensiones (crear índice con una dimensión y subir vectores de otra) | `embeddings.py` centraliza `EMBEDDING_DIMENSION` según el proveedor; `setup_pinecone.py` valida la dimensión del índice existente antes de usarlo |
| Ignorar el namespace | Todo el pipeline usa `PINECONE_NAMESPACE` de forma consistente (ingesta y consulta) |
| Chunks mal dimensionados | `CHUNK_SIZE_TOKENS=650` (punto medio del rango 500-800 recomendado) |
| Reindexar de más | `ingest.py` verifica si el namespace ya tiene vectores antes de volver a subir todo |
| BM25 sin limpiar texto | `rag_system.py` usa un tokenizador propio (regex `\w+`) en vez del default de BM25Retriever (que solo separa por espacios y falla con formato Markdown pegado a las palabras) |

---

## 📊 Configuración

| Variable de entorno | Default | Descripción |
|---|---|---|
| `PINECONE_API_KEY` | - | **Requerida.** Key de tu cuenta de Pinecone |
| `INDEX_NAME` | `entregable-4-rag` | Nombre del índice Serverless |
| `PINECONE_CLOUD` / `PINECONE_REGION` | `aws` / `us-east-1` | Región del índice Serverless |
| `PINECONE_NAMESPACE` | `documentos-tecnicos` | Namespace donde se guardan los vectores |
| `EMBEDDINGS_PROVIDER` | `local` | `local` (fastembed, gratis, dim. 384) u `openai` (dim. 1536) |
| `CHUNK_SIZE_TOKENS` / `CHUNK_OVERLAP_TOKENS` | `650` / `100` | Tamaño y superposición de los chunks |
| `RAG_TOP_K` | `5` | Cantidad de documentos a devolver por consulta |

---

## ✅ Checklist de la consigna

| Requisito | Dónde está |
|---|---|
| Pipeline de ingesta a Pinecone Serverless con metadata avanzada | `ingest.py` (source, categoría, chunk_index, texto completo) |
| Recuperador híbrido (vectorial + BM25) | `rag_system.py` → `RAGSystem` con `EnsembleRetriever` |
| Script de evaluación con Precision@k y Recall@k | `evaluate.py` |
| Índice Serverless con dimensión correcta (1536 si OpenAI) | `embeddings.py` + `setup_pinecone.py` |
| Texto original guardado en metadata (sin BD relacional adicional) | `ingest.py` → campo `"text"` en cada vector |
| `PineconeVectorStore` o SDK nativo de Pinecone | SDK nativo (`pinecone.Pinecone`), envuelto en `PineconeRetriever` |
| `BM25Retriever` + `EnsembleRetriever` | `rag_system.py` |
| Namespace usado consistentemente | `NAMESPACE` en `ingest.py`, reutilizado en `rag_system.py` |
| Chunking 500-800 tokens | `ingest.py` → `CHUNK_SIZE_TOKENS=650` |
| Clase `RAGSystem` con método de búsqueda top-5 | `rag_system.py` → `RAGSystem.buscar()` |
| Golden Set de 5 preguntas | `golden_set.json` |
| Reporte de métricas en consola | `evaluate.py` → `imprimir_reporte()` |
| README con pasos para replicar el índice | Esta sección ("Cómo replicar el índice") |
| Sin API keys en el código (usa `.env`) | Todos los módulos leen de `os.getenv()` |

---

**Listo para escalar a datasets reales en producción. 🚀**
