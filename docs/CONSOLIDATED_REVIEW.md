# Reporte Consolidado - neuro_transformer

**Fecha:** 2026-03-12
**Version actual:** 0.4.0
**Autor:** Revisión automatizada de código

---

## 1. Resumen Ejecutivo

El proyecto **neuro_transformer** implementa un framework de benchmarking para comparar arquitecturas de transformers bio-inspiradas contra baselines estándar en tareas de next-token prediction con datos reales. El proyecto se encuentra en un estado funcional sólido con:

- **27 tests unitarios** pasando al 100%.
- **Modelo DCA entrenado** y exportado (`checkpoints_dca/best_model_export/`).
- **Compliance R1-R5** verificada y completa.
- **Evaluaciones reales ejecutadas**: in-domain (FineWeb-Edu) y out-of-domain (Wikipedia ES).
- **3 releases** documentados (0.2.0, 0.3.0, 0.4.0) con changelogs y release checks.

---

## 2. Estructura del Proyecto

```
neuro_transformer/
├── models/                    # Arquitecturas bio-inspiradas
│   ├── blocks.py              # RMSNorm, RoPE, SwiGLU, MultiHeadAttention
│   ├── base_transformer.py    # Baseline transformer estándar
│   ├── dca.py                 # Dynamic Connectome Architecture (sparse real)
│   ├── mopn.py                # Multi-Orthogonal Processing Networks
│   ├── sct.py                 # Sleep-Cycle Transformers
│   ├── gma_moe.py             # Glial Modulation + Mixture of Experts
│   └── neuro_model.py         # NeuroModel wrapper multicapa
├── neuro_architectures_v2.py  # NeuroModelV2 (PMT + CEN + VLM)
├── data/
│   ├── real_data.py           # Pipeline de datos reales (HF datasets)
│   └── synthetic_data.py      # Datos sintéticos para testing
├── experiments/
│   ├── train_real.py          # Entrenamiento real + export bundle
│   ├── train_loop.py          # Bucle de entrenamiento genérico
│   ├── run_benchmark.py       # Benchmark multi-arquitectura
│   └── eval_dca_real.py       # Evaluación standalone de bundles DCA
├── utils/
│   ├── compliance.py          # Compliance gate R1-R5
│   ├── profiler.py            # DeviceTimer + FLOPs estimation
│   ├── plots.py               # Visualizaciones de benchmark
│   ├── crossover_analysis.py  # Análisis de crossover de scaling laws
│   ├── metrics.py             # (vacío - sin implementar)
│   └── verificar_cuda.py      # Utilidad de verificación CUDA
├── tests/                     # 7 archivos de test, 27 tests
├── docs/                      # Documentación técnica
├── checkpoints_dca/           # Modelo DCA entrenado
└── checkpoints_dca_debug/     # Modelo DCA debug (más pequeño)
```

---

## 3. Arquitecturas Implementadas

| Arquitectura | Archivo | Componentes Clave | Estado |
|---|---|---|---|
| **StandardTransformer** | `models/base_transformer.py` | RoPE + RMSNorm + SwiGLU | ✅ Completo |
| **DCA** (Dynamic Connectome) | `models/dca.py` | `FixedSparseLinear` con `torch.sparse` real (R2) | ✅ Completo |
| **MOPN** (Orthogonal Processing) | `models/mopn.py` | Subespacios ortogonales independientes | ✅ Completo |
| **SCT** (Sleep-Cycle) | `models/sct.py` | Consolidación vigilia/sueño + poda | ✅ Completo |
| **GMA-MoE** (Glial MoE) | `models/gma_moe.py` | Router glial + expertos top-k | ✅ Completo |
| **NeuroModelV2** | `neuro_architectures_v2.py` | PMT (early exit/token) + CEN + VLM | ✅ Completo |

---

## 4. Compliance R1-R5

| Requisito | Descripción | Estado | Evidencia |
|---|---|---|---|
| **R1** | Datos reales para loss oficial | ✅ | FineWeb-Edu + SmolLM tokenizer |
| **R2** | DCA sparse real (no máscara densa) | ✅ | `FixedSparseLinear` con `torch.sparse_coo_tensor` |
| **R3** | PMT early exit por token (masking) | ✅ | `active_mask` por capa en `NeuroModelV2.forward()` |
| **R4** | VLM aislado con `.detach()` | ✅ | `hidden_states.detach()` en `VLM_VicariousStudent` |
| **R5** | Profiling preciso con GPU events | ✅ | `DeviceTimer` con `torch.cuda.Event` |

---

## 5. Estado de Tests

**Total: 27 tests, 27 OK** (ejecutados con `unittest discover`)

| Archivo | Tests | Cobertura |
|---|---|---|
| `test_benchmark_device_selection.py` | 4 | Selección de dispositivo CUDA/CPU |
| `test_crossover_analysis.py` | 4 | Análisis de crossover scaling laws |
| `test_dca_sparse_amp_fallback.py` | 2 | Fallback FP32 para sparse AMP |
| `test_r2_r5_compliance.py` | 7 | Compliance R1-R5 completa |
| `test_scaling_benchmark.py` | 5 | Benchmark de scaling (batch, stacking, grad accum) |
| `test_train_real_export.py` | 2 | Export/load roundtrip de bundles |
| `test_train_real_progress.py` | 2 | Formato de progreso de entrenamiento |

---

## 6. Modelo DCA Entrenado - Resultados

### Configuración del modelo
- **Arquitectura:** `NeuroModelV2ForLM` (vocab=49152, embed=512, layers=12)
- **Dataset:** FineWeb-Edu (sample-10BT)
- **Tokenizer:** SmolLM-135M
- **Entrenamiento:** 3 epochs, seq_len=512, batch=8, PMT threshold=0.85

### Métricas de entrenamiento
- **Train loss:** 5.916
- **Val loss:** 5.923

### Evaluación In-Domain (FineWeb-Edu)
- **Prediction loss:** 5.845
- **Perplexity:** 345.5
- **Throughput:** ~14,886 tokens/s (CUDA, bf16)
- **Capas usadas:** 12.0/12 (avg_depth_ratio ≈ 1.0)

### Evaluación Out-of-Domain (Wikipedia ES)
- **Prediction loss:** 8.488
- **Perplexity:** 4,855.5
- **Throughput:** ~14,926 tokens/s (CUDA, bf16)
- **Capas usadas:** 12.0/12

---

## 7. Mejoras Identificadas y Pendientes

### 🔴 Prioridad Alta

#### 7.1 PMT Early Exit no se activa en la práctica
**Hallazgo crítico:** En ambas evaluaciones, `avg_layers_used ≈ 12.0` y `avg_depth_ratio ≈ 1.0`. El mecanismo PMT no está produciendo early exit efectivo con `threshold=0.85`. Esto significa que el modelo usa las 12 capas para virtualmente todos los tokens, perdiendo la ventaja principal de eficiencia de PMT.

**Acciones sugeridas:**
1. Realizar un sweep de PMT threshold (0.3, 0.5, 0.7, 0.85, 0.95) para encontrar el punto donde empiezan los early exits.
2. Analizar la distribución de confianza por capa para diagnosticar si las early exit heads están sub-entrenadas.
3. Considerar añadir una loss auxiliar que incentive a las early exit heads a producir predicciones confiables en capas tempranas.
4. Evaluar si el `pmt_reward_weight=0.05` es suficiente para incentivar el early exit durante el entrenamiento.

#### 7.2 Perplexity alta (val_ppl ≈ 345)
La perplexity de 345 en FineWeb-Edu es significativamente más alta que modelos de referencia de tamaño comparable (~135M params). Aunque el modelo es más pequeño en parámetros efectivos, hay margen de mejora.

**Acciones sugeridas:**
1. Entrenar por más epochs (actualmente solo 3).
2. Experimentar con learning rate schedules más agresivos.
3. Aumentar `num_samples` de entrenamiento (actualmente 50,000 por defecto).
4. Evaluar si `embed_dim=512` con 12 capas sparse es suficiente para el vocabulario de 49K tokens.

#### 7.3 `utils/metrics.py` está vacío
El archivo existe pero no tiene contenido. Debería contener utilidades de métricas reutilizables.

**Acciones sugeridas:**
1. Implementar funciones para perplexity, accuracy top-k, y métricas de eficiencia.
2. O eliminar el archivo si no se va a usar.

### 🟡 Prioridad Media

#### 7.4 No hay tests para `eval_dca_real.py`
El nuevo script de evaluación standalone no tiene cobertura de tests.

**Acciones sugeridas:**
1. Añadir tests unitarios para `select_eval_device`, `configure_cuda_backend`, `resolve_amp_dtype`.
2. Añadir test de integración con mock de modelo/dataloader.

#### 7.5 No hay tests para `data/real_data.py` ni `data/synthetic_data.py`
Los módulos de datos carecen de tests unitarios dedicados.

**Acciones sugeridas:**
1. Test de `create_real_dataloaders` con mock de datasets HF.
2. Test de `SyntheticNextTokenDataset` verificando shapes, shift correcto, y reproducibilidad con seed.

#### 7.6 `requirements.txt` sin versiones pinneadas
Las dependencias no tienen versiones fijas, lo que puede causar incompatibilidades en nuevos entornos.

**Acciones sugeridas:**
1. Pinnear versiones mínimas: `torch>=2.0`, `transformers>=4.35`, `datasets>=2.14`, etc.
2. Considerar añadir `pyproject.toml` para gestión moderna de dependencias.

#### 7.7 GMA-MoE ejecuta todos los expertos (no hay ahorro real de FLOPs)
Como se documenta en `docs/OPTIMIZATION_REPORT.md`, la implementación actual evalúa todos los expertos y luego aplica pesos de router. No hay ahorro computacional real vs una red densa equivalente.

**Acciones sugeridas:**
1. Implementar sparse dispatch con `torch.scatter/gather` para ejecutar solo top-k expertos por token.
2. O documentar explícitamente esta limitación en el benchmark y ajustar las conclusiones de eficiencia.

#### 7.8 `concept_draft.py` es código legacy
Contiene las implementaciones originales (v0) con máscara densa en DCA, `nn.MultiheadAttention` vanilla, etc. Ya fue superado por las implementaciones en `models/`.

**Acciones sugeridas:**
1. Mover a `docs/` o `archive/` para referencia histórica.
2. O eliminar si no aporta valor.

#### 7.9 CHANGELOG `[Unreleased]` no refleja trabajo reciente
El script `eval_dca_real.py` y las evaluaciones realizadas no están registrados en el changelog.

**Acciones sugeridas:**
1. Registrar en `[Unreleased]`: nuevo script de evaluación standalone, evaluaciones in-domain/out-of-domain, y utilidad `verificar_cuda.py`.

### 🟢 Prioridad Baja

#### 7.10 No hay `conftest.py` en tests
Aunque los tests funcionan, no hay fixtures compartidos ni configuración de pytest.

#### 7.11 No hay CI/CD configurado
No existe `.github/workflows/` ni equivalente para validación automática.

#### 7.12 Falta `__init__.py` en paquetes
Los directorios `models/`, `data/`, `experiments/`, `utils/`, `tests/` no tienen `__init__.py`. Funciona con el path actual pero no es importable como paquete Python estándar.

#### 7.13 Archivos HTML/PHP en la raíz
`index.html` (84KB) y `page-neuro.php` (43KB) parecen ser artefactos de presentación web. Considerar moverlos a un directorio separado (`web/` o `presentation/`).

---

## 8. Mapa de Dependencias entre Módulos

```
experiments/train_real.py
  ├── neuro_architectures_v2.py -> NeuroModelV2ForLM
  │   └── models/dca.py -> SparseDCA_Layer
  │       └── models/blocks.py -> RMSNorm, RoPE, SwiGLU
  ├── data/real_data.py -> create_real_dataloaders
  ├── utils/profiler.py -> DeviceTimer
  └── utils/compliance.py -> build_compliance_report

experiments/eval_dca_real.py
  ├── experiments/train_real.py -> load_exported_model_bundle, evaluate
  └── data/real_data.py -> create_real_dataloaders

experiments/run_benchmark.py
  ├── models/neuro_model.py -> NeuroModel (all architectures)
  ├── data/real_data.py + data/synthetic_data.py
  ├── utils/profiler.py
  ├── utils/compliance.py
  └── utils/plots.py
```

---

## 9. Resumen de Artefactos Generados

| Artefacto | Tamaño | Descripción |
|---|---|---|
| `checkpoints_dca/best_model.pt` | ~4.2 GB | Checkpoint completo (optimizer state incluido) |
| `checkpoints_dca/best_model_export/model.safetensors` | ~1.4 GB | Pesos del modelo exportado |
| `checkpoints_dca/best_model_export/config.json` | 1.3 KB | Config + métricas + compliance |
| `checkpoints_dca/best_model_export/eval_real_metrics_0.json` | 1.2 KB | Eval in-domain FineWeb-Edu |
| `checkpoints_dca/best_model_export/eval_wiki_es.json` | 1.2 KB | Eval out-of-domain Wikipedia ES |
| `checkpoints_dca_debug/` | ~231 MB | Modelo debug más pequeño |

---

## 10. Recomendaciones de Próximos Pasos (Ordenadas por Impacto)

1. **Investigar y resolver la ineficacia del PMT** - Es el hallazgo más crítico. Sin early exit funcional, se pierde la propuesta de valor principal de eficiencia del modelo.
2. **Entrenar por más epochs / más datos** - 3 epochs con 50K muestras es muy poco para un modelo de ~1.4GB. La perplexity de 345 tiene margen de mejora significativo.
3. **Añadir tests para `eval_dca_real.py`** y módulos de datos.
4. **Pinnear versiones en `requirements.txt`**.
5. **Actualizar CHANGELOG** con el trabajo de evaluación reciente.
6. **Registrar como release 0.5.0** cuando se resuelva el tema de PMT y se mejore la perplexity.

---

## 11. Conclusión

El proyecto tiene una **base técnica sólida**: arquitecturas bien implementadas, compliance verificada, tests pasando, documentación completa, y un pipeline reproducible end-to-end. El hallazgo principal es que el mecanismo PMT (early exit por token), que es la innovación central de eficiencia, **no está produciendo ahorro computacional real** con la configuración actual. Resolver esto debería ser la prioridad #1 antes de cualquier claim de eficiencia vs baselines.
