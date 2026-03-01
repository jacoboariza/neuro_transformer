# Diseno funcional y tecnico - neuro_transformer (Req2Design)

Fecha: 2026-02-28
Documento fuente: `docs/REQUIREMENTS.md`
Version del documento fuente: no especificada (se define baseline de trabajo `REQ-v0.1`)

---

## Paso 0 - Fuente de requisitos
- Fuente unica utilizada: `docs/REQUIREMENTS.md`.
- Se mantiene literalidad de R1-R5 sin agregar requisitos externos.
- Gap detectado de gobierno: el documento fuente no tiene version ni fecha de emision; se recomienda agregarlas en el siguiente release.

## Paso 1 - Comprension funcional

### Objetivo del sistema
Construir un framework reproducible que compare calidad y eficiencia entre:
1. Baselines (Transformer estandar + SmolLM-135M local).
2. Arquitecturas bio-inspiradas v1.0 (DCA, MOPN, SCT, GMA-MoE).
3. Modulos cognitivos v2.0 (PMT, CEN, VLM).

Con foco en demostrar mejor relacion precision-eficiencia respecto a LLMs densos.

### Actores implicados
- Investigador/a de IA: define hipotesis, configuraciones y analiza resultados.
- Ingeniero/a ML: implementa, ejecuta benchmarks y valida cumplimiento R1-R5.
- Operador/a de entorno local: prepara GPU/CPU, dependencias y artefactos.

### Flujos principales
- F1. Definir experimento y configuracion reproducible.
- F2. Cargar dataset real y tokenizar con tokenizer oficial de SmolLM.
- F3. Construir topologias y ejecutar entrenamiento/evaluacion.
- F4. Perfilar tiempo/FLOPs de forma precisa (GPU events).
- F5. Generar reporte final (CSV + visualizaciones + ranking).

### Ambiguedades o lagunas detectadas (no bloqueantes)
1. Dataset real objetivo no esta fijado de forma unica (FineWeb-Edu vs WikiText).
2. No se fija umbral objetivo de perplejidad/loss por arquitectura.
3. No se define herramienta exacta de FLOPs (PyTorch profiler vs libreria externa).

## Paso 2 - Alcance y limites

### Que SI hace el sistema
- Benchmark reproducible multi-arquitectura en next-token prediction.
- Entrenamiento y evaluacion con datos reales (R1).
- Comparacion de trade-off entre parametros, coste computacional y loss final.
- Export de resultados en CSV y graficos.

### Que NO hace el sistema
- No realiza serving online ni API de inferencia productiva.
- No sustituye entrenamiento distribuido a gran escala multi-nodo.
- No usa datos sinteticos para reportar loss principal del benchmark oficial.

### Dependencias externas
- Hugging Face datasets/tokenizer/model hub (SmolLM-135M).
- PyTorch (AMP, sparse ops, profiling).
- Librerias de visualizacion/reporting (pandas, matplotlib).

### Supuestos explicitos
- Se dispone de entorno local con recursos minimos de entrenamiento.
- El benchmark se ejecuta offline o con conectividad inicial para descargar artefactos HF.
- El reporte final prioriza comparabilidad bajo protocolo unico.

## Paso 3 - Diseno funcional (sin detalle tecnico)

### Flujo F1 - Configuracion del benchmark
- Entrada: parametros de corrida (dataset, arquitecturas, epochs, seed, precision).
- Proceso: validar parametros, fijar semilla, registrar metadata de experimento.
- Salida: ejecucion inicializada con configuracion trazable.

### Flujo F2 - Preparacion de datos reales
- Entrada: nombre de dataset real + tokenizer oficial SmolLM.
- Proceso: descargar/cargar corpus, tokenizar, segmentar en ventanas autoregresivas, separar train/val.
- Salida: DataLoaders reales listos para entrenamiento y evaluacion.

### Flujo F3 - Entrenamiento/evaluacion de arquitecturas
- Entrada: DataLoaders + arquitectura seleccionada + hiperparametros.
- Proceso: entrenar cada modelo, evaluar loss/perplejidad en validacion, guardar mejor checkpoint.
- Salida: metricas por arquitectura en formato comparable.

### Flujo F4 - Eficiencia computacional
- Entrada: corrida por arquitectura.
- Proceso: medir tiempos de entrenamiento e inferencia con metodo preciso segun hardware; medir FLOPs.
- Salida: metricas de eficiencia confiables y auditables.

### Flujo F5 - Reporte final
- Entrada: metricas de calidad + eficiencia + metadata de corrida.
- Proceso: consolidar resultados, rankear modelos, generar visualizaciones.
- Salida: CSV y graficos finales para analisis cientifico.

## Paso 4 - Diseno logico/tecnico (alto nivel)

### Componentes principales
1. **Experiment Orchestrator**
   - Responsabilidad: coordinar ejecucion end-to-end y reproducibilidad.
   - I/O: recibe `ExperimentConfig`, emite `RunArtifacts`.

2. **Real Data Pipeline**
   - Responsabilidad: cumplir R1 (solo datos reales para loss oficial).
   - I/O: `DatasetSpec -> (train_loader, val_loader, tokenizer)`.

3. **Model Factory**
   - Responsabilidad: instanciar Transformer, SmolLM, DCA/MOPN/SCT/GMA-MoE y NeuroModelV2.
   - I/O: `ModelSpec -> torch.nn.Module`.

4. **Sparse DCA Engine (R2)**
   - Responsabilidad: ejecutar DCA con `torch.sparse` o block-sparse real (sin mascara densa multiplicativa).
   - I/O: `hidden_states -> hidden_states`.

5. **PMT Token Exit Controller (R3)**
   - Responsabilidad: early exit por token via masking (sin `return` por batch/secuencia).
   - I/O: `token_hidden + confidence -> active_token_mask + updated_hidden`.

6. **CEN Branch Simulator**
   - Responsabilidad: simular ramas latentes y elegir ruta coherente en capas objetivo.
   - I/O: `hidden_states -> hidden_states`.

7. **VLM Sidecar Learner (R4)**
   - Responsabilidad: aprendizaje vicario aislado del gradiente maestro.
   - I/O: `hidden_states.detach() -> vlm_loss`.

8. **Trainer + Loss Composer**
   - Responsabilidad: optimizacion con loss compuesta (prediccion + incentivo PMT + vicario).
   - I/O: batch tokenizado -> metricas por step/epoch.

9. **Profiler Service (R5)**
   - Responsabilidad: medir tiempo/FLOPs excluyendo transferencia async.
   - I/O: bloques de computo -> trazas de profiling.

10. **Metrics & Reporting**
    - Responsabilidad: agregacion, ranking, export CSV y visualizaciones.
    - I/O: metricas crudas -> `benchmark_results.csv` + plots.

### Interfaces clave (contratos)
- `create_real_dataloaders(config) -> train_loader, val_loader, tokenizer`
- `build_model(model_spec) -> model`
- `train_one_model(model, loaders, train_config) -> ModelRunMetrics`
- `profile_model(model, batch, profile_config) -> ProfileMetrics`
- `export_report(all_metrics, output_dir) -> csv_path, plot_paths`

### Donde interviene el LLM y para que
- SmolLM-135M local interviene **solo** como baseline de referencia de calidad/eficiencia.
- No interviene para toma de decisiones del framework ni para controlar pipeline.

## Paso 5 - Modelo de datos (alto nivel)

### Entidades
1. **RequirementItem**
   - `id` (R1..R5), `description`, `status`, `evidence`.

2. **ExperimentConfig**
   - `run_id`, `seed`, `dataset_name`, `tokenizer_name`, `seq_len`, `batch_size`, `epochs`, `device`, `amp_dtype`.

3. **ModelSpec**
   - `model_name`, `family` (baseline/v1/v2), `num_layers`, `embed_dim`, `sparse_mode`, `pmt_mode`.

4. **BatchState**
   - `input_ids`, `targets`, `attention_mask`, `active_token_mask`.

5. **TrainingMetrics**
   - `train_loss`, `val_loss`, `perplexity`, `tokens_per_sec`, `params_trainable`.

6. **ProfilingMetrics**
   - `forward_ms`, `backward_ms`, `step_ms`, `flops`, `memory_peak_mb`, `measurement_backend`.

7. **RunArtifact**
   - `checkpoint_path`, `csv_path`, `plots_paths`, `logs_path`.

Relaciones:
- `ExperimentConfig` 1:N `ModelSpec`
- `ModelSpec` 1:N `TrainingMetrics`
- `ModelSpec` 1:N `ProfilingMetrics`
- `ExperimentConfig` 1:N `RunArtifact`

## Paso 6 - Estrategia de IA

### Que tareas hace el LLM
- Inferencia y/o evaluacion como baseline real (SmolLM local).
- Produccion de logits para metricas de calidad comparables.

### Que NO debe hacer el LLM
- No define reglas de negocio del benchmark.
- No reemplaza calculo de metricas ni profiling.
- No decide automaticamente ganadores sin pipeline de scoring reproducible.

### Entradas del LLM
- `input_ids`, `attention_mask`, contexto tokenizado real.

### Salidas del LLM
- `logits`, `loss` (si aplica), metricas derivadas de calidad.

### Modalidad de interaccion
- Sincrona por batch durante entrenamiento/evaluacion.

### Confirmacion de stack local
- El requisito explicito del proyecto fija **SmolLM local via Hugging Face**.
- Por coherencia con requisitos, **no se adopta Ollama** en esta iteracion.

## Paso 7 - Errores y casos limite

1. **Entrada invalida de configuracion**
   - Comportamiento: abortar corrida con mensaje claro y ejemplo de parametro valido.

2. **Fallo al descargar dataset/tokenizer/modelo HF**
   - Comportamiento: reintentos controlados + fallback definido + estado `load_error` trazable.

3. **OOM en GPU / incompatibilidad AMP**
   - Comportamiento: fallback a precision segura o batch menor; registrar degradacion.

4. **Ruta DCA no-sparse accidental**
   - Comportamiento: fallo de compliance (R2) y exclusion del ranking oficial.

5. **PMT sin token masking (exit por return global)**
   - Comportamiento: fallo de compliance (R3), corrida invalida para reporte oficial.

6. **VLM contaminando gradientes del maestro**
   - Comportamiento: test de gradiente falla y bloqueo de merge (R4).

7. **Profiling impreciso en GPU**
   - Comportamiento: si no hay `torch.cuda.Event`, marcar metrica como no valida para comparacion oficial (R5).

## Paso 8 - Criterios de aceptacion (verificables)

### Por flujo funcional
- **AC-F1**: toda corrida exporta `run_id`, seed y config completa en metadatos.
- **AC-F2 (R1)**: loss oficial se calcula solo con DataLoader real; prohibido `torch.randn` en path oficial.
- **AC-F3**: cada topologia produce `train_loss`, `val_loss`, `perplexity`, `params_trainable`.
- **AC-F4 (R5)**: en GPU, tiempos medidos con `torch.cuda.Event(enable_timing=True)` y sincronizacion explicita.
- **AC-F5**: se generan CSV + visualizaciones con ranking reproducible.

### Por componente clave
- **AC-DCA (R2)**: DCA usa kernels sparse reales (`torch.sparse`/block-sparse), sin multiplicacion densa por mascara.
- **AC-PMT (R3)**: PMT aplica enmascaramiento por token; no existe `return` early-exit a nivel batch/secuencia.
- **AC-VLM (R4)**: entrada al estudiante se pasa con `.detach()`; gradientes del maestro permanecen aislados.
- **AC-Reporte**: el ranking final incluye trade-off entre calidad (loss/perplexidad) y eficiencia (tiempo/FLOPs/params).

## Paso 9 - Artefactos de salida

### Artefactos generados en esta ejecucion del workflow
- Diseno funcional y tecnico: `docs/REQ2DESIGN.md` (este documento).
- Lista de decisiones tomadas (abajo).
- Base lista para implementacion dirigida por tareas.

### Decisiones tomadas
1. Documento fuente unico y vinculante: `docs/REQUIREMENTS.md`.
2. En benchmark oficial, la loss valida se calcula solo con datos reales (R1).
3. Se impone compliance gate R2-R5 para que una corrida sea elegible en ranking oficial.
4. SmolLM local via Hugging Face se mantiene como baseline SOTA requerido.

### Listo para `/implement_feature`
1. Implementar DCA sparse real (R2) con ruta `torch.sparse`/BSR.
2. Refactor PMT a token masking (R3) en `NeuroModelV2`.
3. Aislar VLM con `.detach()` + test de no-contaminacion de gradiente (R4).
4. Añadir servicio de profiling con `torch.cuda.Event` + FLOPs (R5).
5. Añadir trazabilidad de compliance y bandera de exclusion en ranking.

---

## Checklist final
- [x] Requisitos comprendidos y sin ambiguedades criticas
- [x] Alcance claramente definido
- [x] Diseno funcional completo
- [x] Diseno tecnico coherente
- [x] Estrategia de IA definida
- [x] Listo para implementacion
