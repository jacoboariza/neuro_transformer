# Release Check Report - v0.2.0

Date: 2026-03-01

## Paso 0 - Contexto del release
- Tipo de release: **minor**
- Rama origen: `main`
- Rama destino: `main`
- Entorno objetivo: `prod`
- Features incluidas:
  - Pipeline de datos reales (`data/real_data.py`)
  - Bloques SOTA (`models/blocks.py`)
  - Wrapper multicapa (`models/neuro_model.py`)
  - Entrenamiento real GPU (`experiments/train_real.py`)
  - Integración `NeuroModelV2` + loss compuesta PMT
  - Gate de compliance R1-R5 + trazabilidad dataset/tokenizer en benchmark
  - Profiling de FLOPs (`torch.profiler`) y timing preciso por dispositivo

## Paso 1 - Versionado
- Versión establecida en archivo `VERSION`: **0.2.0**.
- No hay imágenes Docker ni tags Docker en este repositorio.
- Coherencia de documentación actualizada en `README.md`.

## Paso 2 - Changelog
- Archivo `CHANGELOG.md` actualizado para `0.2.0` con:
  - modernización de capas y sparse kernel en DCA,
  - benchmark oficial sobre datos reales + columnas de provenance,
  - gate de elegibilidad de ranking (`EligibleForRanking`) por R1-R5.

## Paso 3 - Revisión de configuración
- Variables de entorno detectadas:
  - `HF_TOKEN`: opcional para evitar rate limits al descargar modelos del Hub.
- No se detectaron nuevas variables obligatorias para ejecución local.
- Config por defecto mantiene comportamiento seguro.

## Paso 4 - Smoke tests
Comandos ejecutados y resultado:

1. Compilación sintáctica
```powershell
.\.venv\Scripts\python -m py_compile data/real_data.py experiments/run_benchmark.py experiments/train_real.py models/dca.py neuro_architectures_v2.py utils/compliance.py utils/profiler.py utils/plots.py tests/test_r2_r5_compliance.py
```
- Resultado: OK

2. Tests de compliance (R1-R5)
```powershell
.\.venv\Scripts\python -m unittest tests.test_r2_r5_compliance -v
```
- Resultado: OK (`Ran 7 tests ... OK`)

3. Smoke benchmark de provenance + gate de ranking
```powershell
.\.venv\Scripts\python -c "from experiments.run_benchmark import run_benchmark; run_benchmark(epochs=1, batch_size=2, num_samples=32, seq_len=16, embed_dim=32, dataset_name='ag_news', dataset_config=None, tokenizer_name='gpt2', hf_reasoner_model_name='nonexistent/model', output_csv='benchmark_smoke_provenance.csv')"
```
- Resultado: OK. Se exportaron:
  - `benchmark_smoke_provenance.csv`
  - `benchmark_smoke_provenance_ranking.csv`
- Evidencia verificada en CSV:
  - columnas `RequestedDatasetName`, `ResolvedDatasetName`, `RequestedTokenizerName`, `ResolvedTokenizerName`, `UsedFallbackDataset` presentes.
  - `R1_RealData=False` y `EligibleForRanking=False` cuando dataset/tokenizer no son canónicos.
  - `CompositeScore` y `CompositeRank` vacíos en el ranking para runs no elegibles.

4. Smoke train_real (CLI)
```powershell
.\.venv\Scripts\python experiments/train_real.py --help
```
- Resultado: OK

5. Validación de dependencias
```powershell
.\.venv\Scripts\python -m pip check
```
- Resultado: `No broken requirements found.`

## Paso 5 - Validación técnica
- `docker compose up`: **N/A** (no existe `docker-compose.yml` en el repo).
- Warnings críticos de logs: no se observaron.
- Nota: `torch.profiler` emitió warning informativo de ciclo de eventos (no bloqueante).
- Tests automáticos: suite de compliance ejecutada y exitosa (7/7).

## Paso 6 - Seguridad y cumplimiento
- Escaneo básico de secretos (regex común, excluyendo `.venv`) en código fuente del repo: sin hallazgos críticos.
- `pip check` ejecutado:
```powershell
.\.venv\Scripts\python -m pip check
```
- Resultado: `No broken requirements found.`

## Paso 7 - Comunicación del release
### Qué se entrega
- Entrenamiento real migrado a `NeuroModelV2` con incentivo PMT y métricas de eficiencia.
- Infraestructura de datos reales y bloques arquitectónicos modernizados.
- Benchmark oficial con trazabilidad de origen (dataset/tokenizer solicitado vs resuelto).
- Gate de cumplimiento R1-R5 aplicado al ranking oficial.

### Riesgos conocidos
- No hay Docker stack en repo (validación Docker no aplicable).
- Los smoke sobre `fineweb-edu` pueden requerir descargas pesadas del Hub.

### Rollback
1. Volver al commit/tag anterior al release `v0.2.0`.
2. Restaurar `VERSION` y `CHANGELOG.md` a release previa.
3. Reejecutar smoke tests mínimos.

## Paso 8 - Aprobación final
- Reproducible: **Sí** (comandos documentados)
- Reversible: **Sí** (rollback definido)
- Entendible para terceros: **Sí** (README + changelog + reporte)

## Checklist final
- [x] Versión incrementada correctamente
- [x] Changelog actualizado
- [x] Smoke tests ejecutados y documentados
- [x] Docker y configuración validados (N/A: no hay Docker en el repo)
- [x] Sin riesgos críticos conocidos
