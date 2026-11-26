# Hipótesis — LEXICON-X II: Echoes of Eráviri

## Contexto y observaciones de la exploración

- **Corpus:** 1 500 documentos cortos (19–83 caracteres) de 250 sitios, con 6 documentos por sitio:
  - oficial 2048 en `es` y en `er`;
  - oficial 2047 en `es` y en `er`;
  - rumor 2048 (confiabilidad 0.2);
  - archivo 2048 sin cifra (confiabilidad 0.7).
- **Preguntas:** tres tipos.
  - `actual`: la respuesta es el registro oficial de 2048.
  - `historico`: la respuesta es el registro oficial del año pedido.
  - `sin_evidencia`: por ejemplo, «¿Quién dirige ER-X?».
- **Eráviri:** es una lengua artificial (`ER-000 sava 93 ena.`). Ningún modelo preentrenado la conoce, pero el léxico la cubre por completo: 5 objetos más `ena` = disponibles.
- **Las dificultades no están en el significado:**
  - el identificador exacto del sitio (`ER-000` frente a `ER-050`);
  - la vigencia (2048 frente a 2047);
  - la confiabilidad de la fuente (oficial frente a rumor);
  - la decisión de abstenerse.

Para comparar en igualdad de condiciones, ambos sistemas comparten el mismo **módulo de selección y respuesta**:

1. Detectar el sitio, el año y el objeto de la pregunta.
2. Quedarse con los registros oficiales del año objetivo.
3. Extraer la cifra y citar el documento `es` junto con su gemelo `er`.
4. Responder `SIN_EVIDENCIA` si la pregunta no pide una cantidad o si no hay un registro válido.

**Lo único que cambia entre los dos sistemas es el recuperador.**

### Métricas comunes

- Las de `evaluation.py`: EM, `evidence_hit`, `joint` y EM por tipo.
- La latencia media y la p95 por consulta.
- **Recall@k de evidencia:** proporción de preguntas con al menos un `evidence_id` gold entre los *k* primeros documentos recuperados, con k ∈ {1, 3, 6}. Se calcula sin filtro de metadatos para medir solo al recuperador.

**Protocolo:**
- Las reglas se diseñan solo con `train_qa.csv`.
- `validation_qa.csv` se usa una sola vez por sistema, nunca como contexto.
- Las semillas y las versiones de las librerías quedan registradas.

---

## H1 — Modelo estadístico (baseline): BM25 / TF-IDF léxico

**Enunciado.** Un recuperador léxico BM25 con una tokenización que conserva el identificador del sitio como token único (`er-000`) tendrá un Recall@6 de evidencia **≥ 0.95** sin usar filtros de metadatos. Superará en al menos **+10 puntos** al recuperador denso evaluado en las mismas condiciones sin filtro (H2). Con el módulo de selección compartido, alcanzará un `joint` **≥ 0.95** en validación.

**Justificación.**
- Lo que distingue a los documentos de un sitio es el token `ER-###`, que es exacto y poco frecuente: BM25 le asigna un IDF alto.
- Los embeddings densos, en cambio, representan `ER-000` y `ER-050` como vectores casi idénticos.
- En la exploración, la búsqueda densa sin filtro ya devolvió en el top-5 documentos de ER-050 y ER-070 para una pregunta sobre ER-000.

**Diseño experimental.**
- `TfidfVectorizer` (de `scikit-learn`, ya disponible en `requirements.txt`) o BM25 sobre el texto. Los documentos `er` se indexan en dos versiones: el original y una traducción con el léxico.
- `token_pattern` que preserve `ER-\d{3}`.
- Ablación: tokenización por defecto (que separa `er` y `000`) frente a tokenización con el ID completo.

**Criterios de refutación.** H1 se rechaza si ocurre cualquiera de estas condiciones:
- Recall@6 < 0.95.
- La diferencia con el recuperador denso sin filtro es < 10 puntos.
- `joint` < 0.95.
- La ablación de tokenización no cambia el Recall@6 en más de 2 puntos. Esto indicaría que el ID no es el factor decisivo.

---

## H2 — Base de datos vectorial (alternativa): Qdrant + `multilingual-e5-small` + filtros de metadatos

**Enunciado.** La recuperación densa multilingüe en Qdrant, con un filtro de payload por sitio y una expansión de la consulta con el léxico, igualará a H1 en validación (`joint` dentro de ±2 puntos). Será **más robusta frente a paráfrasis**: sobre un conjunto de preguntas parafraseadas derivado de `train_qa.csv`, su EM bajará **≤ 5 puntos** respecto a las preguntas originales, mientras que la de H1 bajará **≥ 15 puntos**. A cambio, su latencia será mayor (al menos ×2 frente a H1).

**Justificación.**
- Los embeddings capturan sinónimos y reformulaciones que no comparten palabras con el corpus, como «existencias», «unidades», «stock», «hoy» o «en este momento», y que BM25 no puede emparejar.
- La debilidad conocida del modelo denso, la exactitud del ID, se cubre con el filtro de metadatos de Qdrant (`site == ER-###`). El resto de los metadatos (año, fuente, confiabilidad) se usan en la selección.

**Diseño experimental.**
- El índice ya está construido en `database/` (`index_corpus.py`) y la consulta en `database/query_rag.ipynb`.
- **Conjunto de parafraseo:** unas 100 preguntas de `train_qa.csv` reescritas con plantillas manuales de sinónimos y reordenamientos. Se mantienen los gold originales. No se usa validación.
- **Ablaciones:**
  1. sin filtro de sitio;
  2. sin traducir los documentos `er` con el léxico;
  3. sin los prefijos `query:`/`passage:` de E5.

**Criterios de refutación.** H2 se rechaza si ocurre cualquiera de estas condiciones:
- Su `joint` en validación queda más de 2 puntos por debajo de H1.
- La caída de EM con paráfrasis es > 5 puntos.
- La caída de H1 con paráfrasis no supera a la de H2 en al menos 10 puntos.
- Al quitar el filtro de sitio, el Recall@6 no baja (lo que contradiría que el ID es su punto débil).

---

## Riesgos y casos difíciles a analizar (≥ 5)

1. **Contradicción vigente frente a anterior:** 2048 (93) frente a 2047 (100). La búsqueda densa puede poner primero el registro de 2047.
2. **Rumor con cifra:** 112, confiabilidad 0.2. No debe ganar aunque tenga el puntaje de similitud más alto.
3. **Ausencia de evidencia:** «¿Quién dirige ER-X?». Ambos recuperadores devuelven documentos, así que la abstención depende de la regla y no del puntaje.
4. **Año sin registro** (por ejemplo, 2045): debe responder `SIN_EVIDENCIA` y no la cifra más cercana.
5. **Pregunta en Eráviri o mixta** («¿Cuántas tavi hay en ER-001?»): pone a prueba la traducción con el léxico en la consulta.
6. **Cita adicional:** se cita el gemelo `er`. Si se cita un documento incorrecto junto al correcto, se discute en el análisis manual.
