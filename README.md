# INF657-final-g4 — LEXICON-X II: Echoes of Eráviri

Asistente de preguntas y respuestas **multilingüe (español / Eráviri)** con evidencia verificable. A partir de una pregunta en lenguaje natural, devuelve una cifra o `SIN_EVIDENCIA`, junto con los IDs de los documentos que la respaldan.

Se comparan dos sistemas que comparten el mismo módulo de respuesta y difieren **solo en el recuperador**:

| Sistema | Recuperador | Respuesta |
|---|---|---|
| **Baseline** (estadístico) | TF-IDF de 1–2 gramas (`scikit-learn`) | reranking por reglas |
| **Alternativa** (base vectorial) | Qdrant + `intfloat/multilingual-e5-small` + filtro de sitio | reranking por reglas |

> **No se usa ningún LLM generativo.** El único modelo neuronal es un *encoder* de embeddings, que convierte texto en vectores y no genera texto. La respuesta es **extractiva**: la cifra se copia de un documento recuperado. El enunciado lo permite: «Se admite generación extractiva o neuronal; no se exige API ni GPU».

---

## Estructura del repositorio

```
.
├── corpus_documentos.csv          # 1 500 documentos (es / er) con metadatos
├── train_qa.csv, validation_qa.csv# 500 y 150 preguntas con respuesta y evidencia gold
├── lexico.csv                     # léxico Eráviri → español
├── evaluation.py                  # métricas oficiales (EM, evidence_hit, joint, by_kind)
├── HIPOTESIS.md                   # hipótesis falsables H1 (estadística) y H2 (vectorial)
├── LEXICON_XII_Colab_vdb.ipynb    # notebook de entrega: baseline vs. alternativa en validación
└── database/
    ├── pyproject.toml, uv.lock    # dependencias (uv)
    ├── index_corpus.py            # crea la base vectorial Qdrant local
    ├── query_rag.ipynb            # consulta en lenguaje natural → JSON estructurado + evaluación
    └── qdrant_data/               # base vectorial persistida (generada, ignorada por git)
```

---

## Instalación y uso

Requiere Python ≥ 3.12 y [uv](https://docs.astral.sh/uv/).

```bash
cd database
uv sync                                   # instala dependencias
uv run index_corpus.py --recreate         # 1) construye la base vectorial
uv run index_corpus.py --query "¿Cuántas semillas hay en ER-000?" --site ER-000   # prueba rápida
uv run jupyter notebook query_rag.ipynb   # 2) consultas y evaluación sobre la base persistida
uv run jupyter notebook ../LEXICON_XII_Colab_vdb.ipynb   # 3) comparación baseline vs. alternativa
```

En **Colab**, `LEXICON_XII_Colab_vdb.ipynb` es autocontenido:
1. Sube los CSV junto al notebook.
2. Descomenta `%pip install -q qdrant-client sentence-transformers` en la primera celda de código.
3. Ejecuta todo.

El notebook crea su propio Qdrant **en memoria**, así que no necesita `qdrant_data/`.

> ⚠️ El Qdrant local admite **un solo cliente por carpeta**. Si `query_rag.ipynb` está abierto con un kernel activo, `index_corpus.py` y otros procesos fallarán con *«Storage folder … is already accessed by another instance»*. Reinicia o cierra ese kernel primero.

---

## Arquitectura

```
                 ┌──────────── INDEXACIÓN (index_corpus.py, una vez) ────────────┐
 corpus_documentos.csv ─► "passage: " + texto ─► multilingual-e5-small ─► vector 384-d
          │                                                                │
          └─► metadatos: doc_id, site, year, source, reliability, ─────────┤
              language, topic                                              ▼
                                                              Qdrant local (coseno)
                                                              punto = vector + payload
                 └──────────────────────────────────────────────────────────┬────┘
                 ┌──────────── CONSULTA (query_rag.ipynb) ──────────────────┼────┐
 "¿Cuántas semillas hay disponibles actualmente en ER-000?"                 │
   │                                                                        │
   ├─► 1. Parseo (regex + léxico): site=ER-000, year=—, item=semillas       │
   │                                                                        │
   ├─► 2. Recuperación: "query: " + pregunta ─► e5 ─► búsqueda coseno ──────┘
   │                    con filtro de payload site == "ER-000" ─► top-10
   │
   ├─► 3. Reranking por reglas: oficial + año objetivo + cifra + objeto
   │      ─► máx(idioma es, confiabilidad, score) ─► cifra + gemelo er
   │
   └─► 4. Salida estructurada (JSON): answer, citations, kind, rationale, evidence, latency_ms
                 └────────────────────────────────────────────────────────────────┘
```

---

## Componentes

### 1. Datos

- **1 500 documentos** de 250 sitios (`ER-000` … `ER-249`), con **6 documentos por sitio**:

  | Documento | Idioma | Año | Fuente | Confiabilidad | Ejemplo |
  |---|---|---|---|---|---|
  | oficial vigente | es | 2048 | oficial | 0.95 | «El registro oficial de ER-000 confirma 93 semillas disponibles.» |
  | gemelo en Eráviri | er | 2048 | oficial | 0.95 | «ER-000 sava 93 ena.» |
  | oficial anterior | es | 2047 | oficial | 0.90 | «El registro anterior de ER-000 indica 100 semillas…» |
  | gemelo anterior | er | 2047 | oficial | 0.90 | «ER-000 sava 100 ena.» |
  | rumor | es | 2048 | rumor | 0.20 | «Un rumor sobre ER-000 menciona 112 semillas…» |
  | archivo | es | 2048 | archivo | 0.70 | «El archivo de ER-000 registra actividades…, sin inventario adicional.» |

- **Tipos de pregunta:** `actual` (registro vigente 2048), `historico` (año pedido) y `sin_evidencia` (por ejemplo, «¿Quién dirige ER-X?»).
- **Eráviri:** es una lengua artificial. Ningún modelo preentrenado la conoce, pero `lexico.csv` la cubre por completo (`sava`=semillas, `tavi`=celdas, `nora`=vehículos, `kiri`=tablillas, `mura`=depósitos, `ena`=disponibles).

### 2. Indexación — `database/index_corpus.py`

- **Unidad de indexación:** un documento equivale a un vector. Los textos tienen entre 19 y 83 caracteres, así que no hace falta partirlos en fragmentos (*chunking*).
- **Modelo de embeddings:** `intfloat/multilingual-e5-small`.
  - Es un *encoder* transformer multilingüe de unos 118 M de parámetros que produce vectores de 384 dimensiones.
  - Corre en CPU y se eligió por ser multilingüe, ligero y bueno en recuperación.
  - E5 se entrenó con prefijos, así que los documentos se codifican como `passage: <texto>` y las preguntas como `query: <texto>`.
  - Los vectores se normalizan, de modo que el producto punto equivale al coseno.
- **Qdrant en modo local:** `QdrantClient(path="database/qdrant_data")` persiste la colección `corpus` en disco.
  - Cada punto guarda el **vector** y un **payload** con todos los metadatos del CSV.
  - En modo local la búsqueda es exacta (compara contra todos los vectores) y tarda unos 20 ms con 1 500 documentos.
  - Los filtros de payload funcionan sin índices explícitos.
- **Opciones del script:** `--recreate` reconstruye la colección; `--query`, `--site` y `-k` permiten hacer una búsqueda de prueba.

### 3. Parseo de la pregunta — `parse_query`

La pregunta es un texto, y de él se extraen **con reglas** (sin modelo) cuatro elementos:

| Campo | Cómo se obtiene | Ejemplo |
|---|---|---|
| `site` | regex `ER-\d{3}` (sin distinguir mayúsculas) | `ER-000` |
| `year` | regex `\b20\d{2}\b` | `2047` |
| `item` | palabras de la pregunta ∩ {semillas, celdas, …}; las palabras en Eráviri se traducen antes con el léxico | `tavi` → `celdas` |
| `asks_quantity` | pistas como `cuánt`, `cantidad`, `inventario`, `número`, `registraba`, `disponibles` | `True` |

A partir de ellos se fija el tipo: `sin_evidencia` si no se pide una cantidad, `historico` si el año es distinto de 2048 y `actual` en otro caso.

### 4. Recuperación

- **Alternativa (vectorial):** búsqueda por coseno en Qdrant de los 10 documentos más parecidos, con un **filtro de payload** `site == <sitio detectado>`.
  - El filtro compensa la debilidad conocida de los embeddings: representan `ER-000` y `ER-050` casi igual.
  - Sin el filtro, el ejemplo de ER-000 devolvía documentos de ER-050 y ER-070.
- **Baseline (estadístico):** TF-IDF con unigramas y bigramas sobre todo el corpus; se toman los 10 primeros por similitud.
  - El bigrama `er 000` es raro y tiene un IDF alto, así que aísla el sitio sin necesidad de filtro.

### 5. Reranking por reglas — `rerank` / `answer`

Es el paso que **decide** la respuesta. El puntaje de similitud solo desempata, y lo que manda son los **metadatos**, siguiendo las reglas del enunciado («los registros oficiales vigentes prevalecen para preguntas actuales; para preguntas históricas se usa el año solicitado»):

1. **Abstención por intención:** si la pregunta no pide una cantidad, la respuesta es `SIN_EVIDENCIA`, sin citas.
2. **Filtrado:** se conservan solo los documentos con `source == "oficial"`, `year ==` año objetivo (por defecto 2048), del sitio pedido, con una cifra en el texto y que mencionen el objeto pedido. Para los documentos en Eráviri, el objeto se compara sobre la traducción.
3. **Abstención por falta de evidencia:** si no queda ningún candidato (por ejemplo, un año 2045 o un objeto que no corresponde al sitio), la respuesta es `SIN_EVIDENCIA`.
4. **Selección:** se elige el máximo según (`language == "es"`, `reliability`, `score`).
5. **Extracción:** se toma la cifra del texto, ignorando el número del sitio.
6. **Citas:** el documento elegido más su **gemelo en Eráviri** (mismo sitio, año y cifra).

Este paso es imprescindible. En el ejemplo de ER-000, Qdrant puso **primero** el registro de 2047 (100) y **segundo** el vigente de 2048 (93); quedarse con el primer resultado habría dado una respuesta incorrecta. Un *cross-encoder* (reranker aprendido) no ayudaría aquí: los documentos correcto e incorrecto son casi idénticos en texto y difieren solo en metadatos como el año y la fuente.

### 6. Salida estructurada — `RAGAnswer`

`query_rag.ipynb` devuelve un `dataclass` que se serializa a JSON:

```json
{
  "question": "¿Cuántas semillas hay disponibles actualmente en ER-000?",
  "answer": "93",
  "citations": ["D000-0", "D000-1"],
  "kind": "actual",
  "site": "ER-000", "year": null, "item": "semillas",
  "rationale": "Registro oficial 2048 (confiabilidad 0.95); se descartan: archivo, registro de otro año, rumor.",
  "evidence": [{"doc_id": "D000-2", "score": 0.918, "language": "es", "year": 2047,
                "source": "oficial", "reliability": 0.9, "text": "...", "translation": null}, "..."],
  "latency_ms": 20.7
}
```

Las predicciones para la evaluación usan el formato de entrega `qid, answer, citations`, con las citas separadas por `|`.

### 7. Evaluación

- **Métricas** (las de `evaluation.py`):
  - **EM:** la respuesta coincide exactamente con la gold.
  - **evidence_hit:** al menos una cita es correcta, o no hay citas cuando la respuesta es `SIN_EVIDENCIA`.
  - **joint:** se cumplen las dos anteriores.
  - Además se reportan el EM por tipo y la latencia media y p95.
- **Protocolo:**
  - Las reglas se diseñaron solo con `train_qa.csv`.
  - Las respuestas de validación se usan únicamente para medir, nunca como contexto ni como reglas por ID.
- **Archivos generados:**
  - `query_rag.ipynb` escribe `database/predicciones_train_dense.csv` y `database/predicciones_validacion_dense.csv`.
  - `LEXICON_XII_Colab_vdb.ipynb` escribe `predicciones_validacion_<sistema>.csv` y `resultados_validacion.csv`, en el formato de `resultados_plantilla.csv`.

---

## Resultados

Validación (150 preguntas), con la misma ejecución para todos los sistemas (`LEXICON_XII_Colab_vdb.ipynb`, CPU Apple Silicon):

| Sistema | EM | evidence_hit | joint | EM actual | EM histórico | EM sin_evidencia | Latencia media |
|---|---|---|---|---|---|---|---|
| Baseline TF-IDF + reglas | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | **0.5 ms** |
| Qdrant + e5 + filtro + reglas | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 22 ms |
| Ablación: Qdrant + e5 **sin** filtro de sitio | 0.933 | 0.933 | 0.933 | 0.88 | 0.98 | 1.00 | 11 ms |

En entrenamiento (500 preguntas, `query_rag.ipynb`), la alternativa vectorial también obtiene 1.0 en todas las métricas, con una latencia media de 22 ms.

**Lectura de los resultados:**
- **H1 se confirma:** TF-IDF alcanza `joint` = 1.0 sin ningún filtro y es unas **45 veces más rápido**.
- **H2, primera parte, se confirma:** la alternativa vectorial **iguala** a la baseline, pero solo gracias al filtro de payload. La ablación sin filtro pierde 6.7 puntos: 10 errores, 9 de ellos en preguntas `actual` del tipo «inventario vigente», donde el sitio correcto no entra en el top-10.
- **H2, robustez a paráfrasis, no se sostiene:** en el conjunto de estrés, ambos sistemas fallan las mismas paráfrasis. El cuello de botella son las reglas, no el recuperador.

> **¿Por qué 1.0?** Las 650 preguntas oficiales salen de **seis plantillas** y las reglas se diseñaron sobre esas plantillas. Un 1.0 indica que las reglas cubren el formato, no que el sistema generalice. Por eso se incluye un conjunto de estrés.

---

## Casos difíciles y limitaciones

Conjunto de estrés escrito a mano, fuera de las plantillas (última sección de `LEXICON_XII_Colab_vdb.ipynb`):

| Caso | Pregunta | Esperado | TF-IDF | Qdrant | Causa del error |
|---|---|---|---|---|---|
| Contradicción 2048 / 2047 / rumor | ¿Cuántas semillas hay disponibles actualmente en ER-000? | 93 | ✅ 93 | ✅ 93 | — |
| Año sin registro | …registraba ER-000 en 2045? | SIN_EVIDENCIA | ✅ | ✅ | — |
| Ausencia de evidencia | ¿Quién dirige ER-000? | SIN_EVIDENCIA | ✅ | ✅ | — |
| Objeto en Eráviri | ¿Cuántas tavi hay en ER-001? | 26 | ✅ 26 | ✅ 26 | — |
| Objeto inexistente | ¿Cuántos habitantes hay en ER-000? | SIN_EVIDENCIA | ❌ 93 | ❌ 93 | Un objeto desconocido se toma como «cualquier objeto», así que **inventa** una respuesta |
| Fecha relativa | …había en ER-000 el año pasado? | 100 | ❌ 93 | ❌ 93 | No se interpretan expresiones como «el año pasado» |
| Sitio sin formato | ¿Cuántas semillas hay en el sitio 0? | 93 | ❌ 39 | ❌ 23 | La regex `ER-\d{3}` no lo detecta, así que se devuelve otro sitio |
| Sin sitio | ¿Cuántas semillas hay disponibles? | SIN_EVIDENCIA | ❌ 39 | ❌ 41 | No se abstiene: elige un sitio cualquiera |
| Paráfrasis | ¿Cuál es el stock de semillas de ER-000 hoy? | 93 | ❌ | ❌ | «stock» no está entre las pistas de cantidad, así que se abstiene sin motivo |

Aciertos en el conjunto de estrés: TF-IDF 0.44, Qdrant 0.44 y Qdrant sin filtro 0.33.

**Mejoras posibles:**
1. Abstenerse si el objeto no se reconoce o si no hay sitio.
2. Una regex de sitio más tolerante (`ER[\s-]?\d{1,3}`), con normalización y validación contra los 250 sitios.
3. Resolver fechas relativas («el año pasado» → 2047).
4. Ampliar las pistas de cantidad, o reemplazarlas por un clasificador de intención.

**Citas adicionales:** se cita el gemelo en Eráviri, que forma parte de la evidencia gold. `evidence_hit` exige solo una cita correcta y no penaliza las citas incorrectas adicionales, así que esos casos se revisan a mano.

---

## Reproducibilidad

| Componente | Versión |
|---|---|
| Python | 3.12 |
| qdrant-client | 1.20.0 |
| sentence-transformers | 6.3.0 |
| torch | 2.15.0 |
| transformers | 5.24.1 |
| scikit-learn | 1.9.1 |
| pandas / numpy | 3.1.1 / 2.5.5 |
| Modelo de embeddings | `intfloat/multilingual-e5-small` (Hugging Face) |

- Las versiones exactas quedan fijadas en `database/uv.lock`.
- La semilla es `SEED = 42`. El pipeline es determinista: no hay entrenamiento ni muestreo, y la búsqueda local en Qdrant es exacta.
- Las dos últimas celdas de `LEXICON_XII_Colab_vdb.ipynb` (MiniLM y `flan-t5-small`) son recursos **opcionales** de la plantilla y no intervienen en las predicciones.
