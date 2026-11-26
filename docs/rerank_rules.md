# Reglas del reranking (`rerank`)

Este documento describe el módulo de selección compartido por **ambos sistemas** (baseline léxico y alternativa vectorial) en `LEXICON_XII_Colab_vdb.ipynb`. Ninguno de los dos sistemas usa un LLM generativo: el recuperador devuelve los 10 primeros documentos (`(indice_doc, score)`) y estas reglas deciden la respuesta y las citas.

```python
def predict(question, retriever):
    return rerank(parse_query(question), retriever(question))
```

Los dos sistemas difieren **solo** en `retriever` (`tfidf_hits`, `vdb_hits` o `vdb_nofilter_hits`). Así, cualquier diferencia de resultados se debe a la recuperación.

---

## 0. Parseo de la pregunta (`parse_query`)

| Campo | Cómo se obtiene | Ejemplo |
|---|---|---|
| `site` | regex `ER-\d{3}` (sin distinguir mayúsculas) | `ER-000` |
| `year` | regex `\b20\d{2}\b` | `2047` |
| `item` | palabras de la pregunta ∩ {semillas, celdas, vehículos, tablillas, depósitos}; las palabras en Eráviri se traducen antes con `lexico.csv` | `tavi` → `celdas` |
| `asks_quantity` | regex `cu[aá]nt\|cantidad\|inventario\|n[uú]mero\|registraba\|disponibles` | `True` |

---

## 1. Abstención por intención

Si la pregunta **no pide una cantidad** (`asks_quantity == False`), la respuesta es **`SIN_EVIDENCIA`**, sin citas.

> «¿Quién dirige ER-005?» → `SIN_EVIDENCIA`

## 2. Filtrado de candidatos

Un documento recuperado sobrevive solo si cumple **las cinco** condiciones:

| # | Condición | Qué descarta |
|---|---|---|
| a | `source == "oficial"` | rumores (confiabilidad 0.20) y archivos |
| b | `year ==` año objetivo (el de la pregunta; si no hay, **2048**) | registros de otros años |
| c | `site ==` sitio de la pregunta (si la pregunta no trae sitio: cualquier sitio devuelto por el recuperador) | documentos de otros sitios |
| d | el texto contiene un número, sin contar el del sitio | archivos «sin inventario adicional» |
| e | el texto menciona el objeto pedido; los documentos en Eráviri se comparan sobre su traducción | registros de otro objeto |

## 3. Abstención por falta de evidencia

Si no queda ningún candidato, la respuesta es **`SIN_EVIDENCIA`**, sin citas.

> «¿Cuántas semillas registraba ER-000 en 2045?» → no hay registro de 2045 → `SIN_EVIDENCIA`

## 4. Selección del mejor candidato

Se toma el máximo según esta clave, en orden de prioridad:

1. `language == "es"` (español primero)
2. `reliability` (mayor confiabilidad)
3. `score` del recuperador (solo desempata)

## 5. Extracción de la cifra (`number_in`)

Se elimina `ER-\d+` del texto y se toma el primer número (`\b\d+\b`). La cifra se **copia** del documento, no se genera.

## 6. Citas

Se citan el documento elegido y todos los candidatos del **mismo sitio** con la **misma cifra**; en la práctica, su gemelo en Eráviri. El documento elegido va primero.

---

## Ejemplo completo

Pregunta: «¿Cuántas semillas hay disponibles actualmente en ER-000?»
→ `site=ER-000`, `year=None` (se usa 2048), `item=semillas`, `asks_quantity=True`

| Documento | oficial | 2048 | semillas | ¿Sobrevive? |
|---|---|---|---|---|
| «El registro oficial de ER-000 confirma **93** semillas disponibles.» | ✅ | ✅ | ✅ | ✅ elegido (es) |
| «ER-000 sava **93** ena.» (`sava` → semillas) | ✅ | ✅ | ✅ | ✅ gemelo citado |
| «El registro anterior de ER-000 indica **100** semillas…» | ✅ | ❌ 2047 | | ❌ |
| «Un rumor sobre ER-000 menciona **112** semillas…» | ❌ rumor | | | ❌ |
| «El archivo de ER-000 registra actividades…» | ❌ archivo | | | ❌ |

Salida: `answer = 93`, `citations = <doc oficial es>|<gemelo er>`

---

## Relación con el enunciado

- Las reglas **2a + 2b** implementan: «los registros oficiales vigentes prevalecen para preguntas actuales; para preguntas históricas se usa el año solicitado».
- Las reglas **1 y 3** implementan: «no inventen respuestas cuando falte evidencia».
- La regla **5** garantiza que la cifra y la cita coinciden: si EM acierta, evidence_hit también acierta, de modo que `joint ≈ EM`.

## Limitaciones conocidas

- **Regla 1:** la lista de pistas de cantidad es fija. «¿Qué stock tiene ER-000?» da `SIN_EVIDENCIA` aunque la respuesta exista.
- **Regla 2e:** si la pregunta no menciona un objeto reconocido (`item = None`), el filtro no se aplica y se acepta cualquier objeto. Una mejora posible es abstenerse en ese caso.
- **Regla 2c:** si la pregunta no trae sitio, se aceptan todos los sitios del top-10, lo que puede producir respuestas de un sitio equivocado.
- **Paráfrasis:** el cuello de botella son estas reglas, no el recuperador. En el conjunto de estrés ambos sistemas fallan las mismas paráfrasis.
