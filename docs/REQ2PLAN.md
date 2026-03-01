# Plan tecnico ejecutable - neuro_transformer (Req2Plan)

Fecha: 2026-02-28
Fuente principal: `docs/REQ2DESIGN.md`
Fuente de requisitos: `docs/REQUIREMENTS.md`

---

## Paso 1 - Entender el objetivo

### Resumen en 10 bullets
1. El sistema debe comparar baselines (Transformer + SmolLM local) vs arquitecturas bio-inspiradas.
2. El benchmark debe priorizar evidencia empirica reproducible de calidad vs eficiencia.
3. La loss oficial del benchmark debe venir de datos reales tokenizados con tokenizer oficial SmolLM (R1).
4. DCA debe usar sparsity computacional real (`torch.sparse` o block-sparse), no mascara densa multiplicativa (R2).
5. PMT debe operar con early exit a nivel token via masking, sin `return` global por batch o secuencia (R3).
6. VLM debe estar aislado del gradiente del maestro con `detach()` explicito (R4).
7. El profiling en GPU debe medirse con eventos nativos (`torch.cuda.Event`) para tiempos confiables (R5).
8. El resultado final debe incluir CSV y visualizaciones con trade-off params/FLOPs-tiempo/loss.
9. La comparativa debe mantenerse consistente en protocolo de entrenamiento/evaluacion entre modelos.
10. Se requiere un gate de compliance para excluir corridas no conformes de ranking oficial.

### Huecos detectados y preguntas minimas
- Decision cerrada Q1: dataset canonico oficial `HuggingFaceFW/fineweb-edu` (fallback controlado permitido solo como contingencia).
- Decision cerrada Q2: backend oficial de FLOPs `torch.profiler`.

---

## Paso 2 - Propuesta tecnica

### Estado actual vs requisitos
- R1: parcialmente cubierto en `experiments/train_real.py` (datos reales), pero `experiments/run_benchmark.py` sigue entrenando con sinteticos para tabla principal.
- R2: incumplido en `models/dca.py` y `neuro_architectures_v2.py` (mascara densa multiplicativa).
- R3: incumplido en `neuro_architectures_v2.py` (early return global por secuencia).
- R4: incompleto; existe `torch.no_grad()` en VLM, pero se exige `detach()` explicito en la entrada al estudiante.
- R5: incumplido; tiempos con `time.time()` sin eventos GPU.

### Arquitectura objetivo (componentes)
1. **ComplianceGate**
   - Evalua R1-R5 por corrida y marca `eligible_for_ranking`.
2. **RealDataPipeline**
   - Unifica carga/tokenizacion para benchmark oficial.
3. **SparseDCAKernel**
   - DCA con pesos sparse reales (`torch.sparse_coo`/BSR).
4. **TokenMaskingPMTController**
   - Mantiene `active_token_mask` por capa; actualiza solo tokens sorprendidos.
5. **VLMIsolatedSidecar**
   - Consume `hidden.detach()` y calcula `vlm_loss` aislado.
6. **ProfilerService**
   - Backend CPU (`perf_counter`) y backend GPU (`cuda.Event`).
7. **BenchmarkOrchestratorV2**
   - Ejecuta modelos bajo protocolo unico real-data + profiling.
8. **ReportingService**
   - Exporta CSV/plots + columnas de compliance + FLOPs.

### Datos e interfaces (contratos)
- `prepare_real_benchmark_loaders(cfg) -> train_loader, val_loader, tokenizer`
- `build_sparse_dca_layer(cfg) -> nn.Module`
- `forward_with_token_mask(hidden, mask, threshold) -> hidden, mask, token_logits`
- `compute_vlm_loss(hidden_detached) -> Tensor`
- `profile_step(step_fn, device) -> {forward_ms, backward_ms, step_ms, flops}`
- `evaluate_compliance(run_artifact) -> {R1:bool,...,R5:bool, eligible_for_ranking:bool}`

### Decisiones y trade-offs
1. **Sparsity real primero en DCA (R2)**
   - Pro: impacto directo en FLOPs/VRAM.
   - Contra: mayor complejidad de kernel y debugging.
2. **Token masking incremental para PMT (R3)**
   - Pro: cumple requisito y preserva batch throughput.
   - Contra: manejo de mascaras y estados por capa.
3. **`detach()` + sidecar opt-in para VLM (R4)**
   - Pro: aislamiento matematico verificable.
   - Contra: posible menor senal de aprendizaje vicario.
4. **Profiler nativo PyTorch (R5)**
   - Pro: sin nuevas dependencias iniciales.
   - Contra: FLOPs puede requerir pasos extra de configuracion.

---

## Paso 3 - Plan ejecutable

### Orden de implementacion (hitos)
1. H0 - Infraestructura de pruebas y compliance gate.
2. H1 - R2 (DCA sparse real).
3. H2 - R3 (PMT token masking).
4. H3 - R4 (VLM detach + pruebas de gradiente).
5. H4 - R5 (profiling GPU con eventos).
6. H5 - Integracion benchmark oficial real-data + reporte final.

### Backlog granular (1-3h por tarea)

| ID | Tarea | Est. | Dependencias | Entregable |
|---|---|---:|---|---|
| T01 | Crear `tests/` base + `conftest.py` + util de seeds | 1h | - | Estructura de tests reproducible |
| T02 | Crear `utils/compliance.py` con estructura R1-R5 | 2h | T01 | Evaluador de compliance por corrida |
| T03 | Instrumentar `experiments/train_real.py` para exportar evidencia R1-R5 | 2h | T02 | Metadata de compliance en checkpoint/log |
| T04 | Implementar `models/dca_sparse.py` con pesos sparse reales | 3h | T01 | DCA sparse funcional |
| T05 | Conectar DCA sparse en factory/model wrappers | 2h | T04 | Ruta de modelo seleccionable |
| T06 | Test unitario: asegurar no uso de mascara densa en ruta R2 | 2h | T04 | Test que falla si hay multiply denso |
| T07 | Refactor `NeuroModelV2` a PMT por token mask (sin early return global) | 3h | T01 | Forward con `active_token_mask` |
| T08 | Ajustar loss PMT para agregacion por token activo/inactivo | 2h | T07 | Loss consistente con masking |
| T09 | Tests PMT: invariantes de mascara y profundidad efectiva por token | 2h | T07 | Cobertura funcional R3 |
| T10 | Refactor VLM para usar `hidden.detach()` explicito | 1h | T01 | VLM aislado de maestro |
| T11 | Test de gradiente: cero gradiente cruzado maestro<-VLM | 2h | T10 | Evidencia R4 |
| T12 | Crear `utils/profiler.py` con backend CPU/GPU (cuda.Event) | 3h | T01 | Servicio de profiling |
| T13 | Integrar profiling en `train_real.py` y `run_benchmark.py` | 2h | T12 | Metricas `forward_ms/backward_ms/step_ms` |
| T14 | Agregar FLOPs estimado en reporte (torch.profiler) | 2h | T12 | Campo FLOPs en CSV |
| T15 | Migrar `run_benchmark.py` a path oficial real-data para ranking | 3h | T03 | Ranking oficial sin loss sintetica |
| T16 | Integrar ComplianceGate en ranking (`eligible_for_ranking`) | 2h | T02,T15 | Exclusion automatica de corridas no conformes |
| T17 | Actualizar `utils/plots.py` con ejes FLOPs/tiempo/loss | 2h | T14,T15 | Visualizaciones finales |
| T18 | Actualizar docs (`README.md`, `docs/REQ2DESIGN.md`, `CHANGELOG.md`) | 2h | T16,T17 | Documentacion alineada |
| T19 | Smoke final end-to-end en CPU y GPU (si disponible) | 2h | T18 | Evidencia de validacion |

### Tests por hito

#### H0 (T01-T03)
- Test de schema de metadata de compliance.
- Test de persistencia de flags R1-R5 en artefactos.

#### H1 (T04-T06)
- Unit test de forward DCA sparse (shape + numerica basica).
- Test de densidad efectiva (`nnz / total`) dentro de umbral esperado.
- Test de regresion para detectar fallback accidental a ruta densa.

#### H2 (T07-T09)
- Test de no existencia de early `return` global en forward PMT.
- Test de token masking: tokens confiables permanecen congelados.
- Test de consistencia de logits para tokens activos.

#### H3 (T10-T11)
- Test de `detach()` explicito antes del sidecar VLM.
- Test de gradiente: parametros del maestro no reciben gradiente desde loss vicaria.

#### H4 (T12-T14)
- Test backend CPU profiler.
- Test backend GPU profiler con `torch.cuda.Event` (si CUDA disponible).
- Test de presencia de columnas de profiling + FLOPs en CSV.

#### H5 (T15-T19)
- Smoke benchmark real-data + ranking con compliance.
- Validacion de visualizaciones y consistencia de metricas.
- Test manual de reproducibilidad con seed fija.

---

## Paso 4 - Validacion

### Criterios de aceptacion verificables
- AC1 (R1): ranking oficial usa unicamente DataLoader real; path sintetico queda marcado como no-oficial.
- AC2 (R2): DCA oficial ejecuta sparse real y no contiene multiplicacion densa por mascara en ruta principal.
- AC3 (R3): PMT opera por token masking; no hay corte global por batch/secuencia.
- AC4 (R4): VLM recibe `hidden.detach()` y pruebas de gradiente pasan.
- AC5 (R5): en GPU los tiempos reportados se obtienen con `torch.cuda.Event(enable_timing=True)`.
- AC6: CSV final incluye `TrainableParams`, `FLOPs`, `Train/InferTime`, `Loss/Perplexity`, `eligible_for_ranking`.

### Riesgos y mitigaciones

| Riesgo | Impacto | Mitigacion |
|---|---|---|
| Complejidad de sparse kernels (R2) | Alto | Empezar con COO/CSR estable y luego optimizar a BSR |
| Overhead de token masking (R3) | Medio | Vectorizar operaciones y evitar bucles Python por token |
| Inestabilidad de entrenamiento tras cambios PMT/VLM | Medio | Fase de ablation y ajuste gradual de pesos de loss |
| Profiling inconsistente entre CPU/GPU | Medio | API unificada con backend explicito y tests separados |
| Dependencia de descarga HF | Medio | Cache local + fallback + reintentos controlados |

### Definition of Done del plan
- Todas las tareas T01-T19 completadas.
- Criterios AC1-AC6 verificados (test o evidencia documentada).
- `docs/release_check` actualizado con resultados de compliance y profiling.

---

## Salida del workflow
- Plan tecnico ejecutable listo para `/04-implement_feature`.
- Backlog granular (1-3h) con orden, dependencias y estrategia de pruebas.
