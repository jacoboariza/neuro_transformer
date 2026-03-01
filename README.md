# Neuro-Transformer Benchmark: Arquitecturas Bio-Inspiradas vs Transformers Estándar y LLMs Locales

Este repositorio implementa un **framework experimental reproducible** para comparar, de forma empírica, una arquitectura Transformer estándar y un modelo local real (Hugging Face SmolLM-135M) frente a cuatro propuestas bio-inspiradas: **DCA, MOPN, SCT y GMA-MoE**.

El proyecto nace de una pregunta de investigación central: **¿podemos reducir costos computacionales y mejorar eficiencia estructural más allá del patrón de atención densa de los Transformers clásicos, sin perder capacidad de modelado secuencial?** Para ello, se exploran mecanismos inspirados en principios neurobiológicos (conectividad escasa, subprocesamiento ortogonal, consolidación por ciclos y modulación glial) y se evalúan bajo un protocolo común de benchmark.

---

## 1) Arquitecturas implementadas

### 1.1 Baseline Transformer
- **Archivo:** `models/base_transformer.py`
- Capa Transformer modernizada con:
  - atención causal con **RoPE** (Rotary Positional Embeddings)
  - bloque feed-forward **SwiGLU**
  - normalización **RMSNorm** + residual

### 1.2 DCA (Dynamic Connectome Architecture)
- **Archivo:** `models/dca.py`
- Implementa enrutamiento escaso real con kernels `torch.sparse` (COO) en la ruta principal.
- Inspiración: conectoma biológico con conectividad no densa.

### 1.3 MOPN (Multi-dimensional Orthogonal Processing Networks)
- **Archivo:** `models/mopn.py`
- Divide embedding en subespacios ortogonales y aplica subredes independientes por subespacio.
- Inspiración: procesamiento paralelo especializado por dimensión funcional.

### 1.4 SCT (Sleep-Cycled Transformers)
- **Archivo:** `models/sct.py`
- Introduce memoria de corto plazo + fase de consolidación `sleep_cycle()`.
- El ciclo de sueño consolida pesos, poda conexiones débiles y resetea memoria temporal.

### 1.5 GMA-MoE (Glial Modulation & Mixture of Experts)
- **Archivo:** `models/gma_moe.py`
- Red glial auxiliar estima complejidad y activa dinámicamente 1..N expertos.
- Inspiración: modulación metabólica adaptativa del cómputo.

### 1.6 Modelos locales reales (Hugging Face)
- **Integración en:** `experiments/run_benchmark.py`
- Función `load_local_reasoner(...)` para cargar:
  - `HuggingFaceTB/SmolLM-135M`
- Se incluye en la misma tabla comparativa junto a las arquitecturas del repositorio.

### 1.7 Bloques SOTA reutilizables
- **Archivo:** `models/blocks.py`
- Componentes base para escalar arquitecturas:
  - `RMSNorm`
  - `RotaryEmbedding`
  - `SwiGLUFeedForward`
  - `MultiHeadSelfAttentionRoPE`

### 1.8 Wrapper multicapa `NeuroModel`
- **Archivo:** `models/neuro_model.py`
- Permite apilar `num_layers` de cualquier bloque (`transformer`, `dca`, `mopn`, `sct`, `gma_moe`) con embedding y proyección final a vocabulario.

### 1.9 Pipeline de datos reales + entrenamiento GPU
- **Datos reales:** `data/real_data.py`
- **Entrenamiento de alto rendimiento:** `experiments/train_real.py`
- Incluye:
  - descarga de subset real (FineWeb-Edu con fallback a Wikipedia ES)
  - tokenización Hugging Face
  - `DataLoader` autoregresivo `(x, y)` con desplazamiento de 1 token
  - AMP (`autocast` + `GradScaler`), `AdamW`, cosine warmup scheduler y checkpoints por mejora de validación
  - entrenamiento con `NeuroModelV2` + pérdida compuesta con incentivo PMT para salida temprana

### 1.10 Prototipo experimental v2
- **Archivo:** `neuro_architectures_v2.py`
- Implementa PMT (Early Exit), CEN (simulación contrafáctica) y VLM (aprendizaje vicario).

---

## 2) Estructura del proyecto

```text
neuro_transformer/
├─ models/
│  ├─ blocks.py             # RoPE, RMSNorm, SwiGLU y atención causal
│  ├─ base_transformer.py   # Baseline Transformer layer
│  ├─ dca.py                # Dynamic Connectome Architecture
│  ├─ mopn.py               # Multi-dimensional Orthogonal Processing Networks
│  ├─ sct.py                # Sleep-Cycled Transformers
│  ├─ gma_moe.py            # Glial Modulation & Mixture of Experts
│  └─ neuro_model.py        # Wrapper multicapa para entrenamiento fundacional
├─ data/
│  ├─ synthetic_data.py     # Dataset y DataLoader sintéticos para next-token prediction
│  └─ real_data.py          # Dataset y DataLoaders de texto real tokenizado
├─ experiments/
│  ├─ train_loop.py         # Función train_model(...) y utilidades de entrenamiento
│  ├─ run_benchmark.py      # Script maestro de benchmark + integración SmolLM
│  └─ train_real.py         # Entrenamiento real con AMP, AdamW, scheduler y checkpoints
├─ neuro_architectures_v2.py # Arquitectura experimental NeuroModelV2 + PMT/CEN/VLM
├─ utils/
│  ├─ compliance.py         # Gate de cumplimiento R1..R5 + elegibilidad de ranking
│  ├─ profiler.py           # Timer por dispositivo + estimación FLOPs con torch.profiler
│  └─ plots.py              # Generación de gráficos desde benchmark_results.csv
├─ tests/
│  └─ test_r2_r5_compliance.py # Tests de cumplimiento técnico (R2-R5 + gate R1)
├─ benchmark_results.csv    # Resultados tabulares exportados por el benchmark
├─ benchmark_results_ranking.csv # Ranking compuesto con gate de compliance
├─ benchmark_params.png     # Gráfico de barras de parámetros entrenables
├─ benchmark_time.png       # Gráfico de barras de tiempos de ejecución
└─ requirements.txt         # Dependencias del proyecto
```

---

## 3) Instalación y requisitos

### 3.1 Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd neuro_transformer
```

### 3.2 Crear y activar entorno virtual

#### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Linux / macOS
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3.3 Instalar dependencias

```bash
pip install -r requirements.txt
```

Dependencias principales:
- `torch`
- `transformers`
- `datasets`
- `tiktoken`
- `safetensors`
- `pandas`
- `matplotlib`

> Nota: La primera ejecución con `HuggingFaceTB/SmolLM-135M` descargará pesos del Hub de Hugging Face. Para mayor estabilidad/rate limit, puede configurarse `HF_TOKEN`.

---

## 4) Uso y reproducibilidad

### 4.1 Ejecutar benchmark completo

```bash
python experiments/run_benchmark.py
```

Este comando:
1. Carga datos reales desde `HuggingFaceFW/fineweb-edu` (fallback controlado) y tokeniza con `HuggingFaceTB/SmolLM-135M`.
2. Entrena y compara:
   - Transformer baseline
   - DCA
   - MOPN
   - SCT
   - GMA-MoE
   - SmolLM-135M (local, vía Hugging Face)
3. Exporta resultados en CSV + ranking con columnas de compliance (`R1..R5`, `EligibleForRanking`), métricas de FLOPs y trazabilidad de origen (`RequestedDatasetName`, `ResolvedDatasetName`, `RequestedTokenizerName`, `ResolvedTokenizerName`, `UsedFallbackDataset`).

Si se activa el fallback de dataset (o se usa un tokenizer distinto al canónico), el benchmark seguirá ejecutándose para diagnóstico, pero quedará marcado como no elegible para ranking oficial (`R1_RealData=False`).

> Nota: Para ejecutar este benchmark oficial deben estar instaladas `datasets` y `transformers` en el entorno.

### 4.2 Entrenar modelo multicapa en datos reales

```bash
python experiments/train_real.py \
  --num-layers 12 \
  --embed-dim 512 \
  --seq-len 512 \
  --batch-size 8 \
  --epochs 5 \
  --pmt-exit-threshold 0.90 \
  --pmt-reward-weight 0.02 \
  --vicarious-loss-weight 0.005
```

Ejemplo rápido (smoke local):

```bash
python experiments/train_real.py --num-samples 2000 --seq-len 128 --batch-size 4 --epochs 1
```

### 4.3 Generar gráficos

```bash
python utils/plots.py
```

### 4.4 Archivos de salida

Tras la ejecución, se generan en la raíz del proyecto:
- `benchmark_results.csv`
- `benchmark_results_ranking.csv`
- `benchmark_params.png`
- `benchmark_time.png`
- `benchmark_flops.png`
- `benchmark_tokens_per_sec.png`
- `benchmark_real_case_accuracy.png`
- `benchmark_real_case_loss.png`
- `benchmark_compliance.png`

El ranking compuesto aplica gate de compliance: cuando `EligibleForRanking=False`, las columnas `CompositeScore` y `CompositeRank` quedan vacías para ese modelo.

En entrenamiento real:
- `checkpoints/best_model.pt`

El checkpoint incluye metadata de reproducibilidad/compliance: dataset solicitado vs resuelto, tokenizer solicitado vs resuelto, backend de profiling y flags `R1..R5`.

---

## 5) Resultados preliminares (placeholder)

El pipeline de benchmarking registra automáticamente métricas comparables por modelo, incluyendo:
- **Pérdida final (`FinalLoss`)**
- **Número de parámetros entrenables (`TrainableParams`)**
- **Tiempo de ejecución total (`TrainTimeSeconds`)**
- **Estado de ejecución (`Status`)**

Este marco puede extenderse para separar explícitamente métricas de inferencia y entrenamiento en experimentos posteriores del paper.

---

## 6) Contexto científico y motivación

Los Transformers han demostrado rendimiento sobresaliente, pero presentan limitaciones relevantes en escalabilidad de cómputo y costo energético, especialmente en escenarios de atención densa. Este proyecto explora, de manera controlada y reproducible, si mecanismos inspirados en organización neural biológica pueden ofrecer trayectorias alternativas de diseño arquitectónico.

Las cuatro propuestas bio-inspiradas no buscan reemplazar de forma inmediata a los LLMs dominantes, sino aportar evidencia empírica sobre estrategias de eficiencia estructural y asignación dinámica de recursos computacionales.

---

## 7) Licencia

Este proyecto se distribuye bajo licencia **MIT**.

```text
MIT License

Copyright (c) [YEAR] [AUTHORS]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 8) Contacto y autores

- **Autores del paper:** Luis Jacobo Ariza Jiménez
- **Afiliación:** www.jacoboariza.com
- **Contacto:** luis.jacobo@gmail.com

Si utilizas este repositorio en investigación académica, se recomienda incluir cita al trabajo final una vez publicado.
