# Reporte de Optimización de Código para GPU

**Fecha:** 1 de Marzo de 2026
**Contexto:** Revisión de eficiencia para entrenamiento y benchmarking de arquitecturas neuro-inspiradas.

## Optimizaciones Implementadas

### 1. Aceleración de Capas Dispersas (`models/dca.py`)
**Problema:** La capa `FixedSparseLinear` reconstruía y coalescía la matriz dispersa en cada paso `forward`, consumiendo ciclos de CPU y ancho de banda innecesario.
**Solución:**
- **Ordenamiento Canónico:** Se ordenan los índices aleatorios al inicializar. Esto facilita la creación de la matriz dispersa en un formato ya compatible con CSR/COO optimizado.
- **Cacheo Inteligente:** Se implementó un mecanismo de caché para `sparse_weight`. En modo `eval()`, la matriz dispersa coalescida se reutiliza entre llamadas, eliminando la sobrecarga de construcción. La caché se invalida automáticamente al cambiar a `train()`.

### 2. Eliminación de Cuellos de Botella en Evaluación (`experiments/`)
**Problema:** Los bucles de validación y benchmarking acumulaban pérdidas y métricas transfiriendo escalares a CPU en cada iteración (`item()`), forzando una sincronización GPU-CPU por cada lote.
**Solución:**
- **Acumulación en GPU:** Se modificaron `train_real.py` y `run_benchmark.py` para acumular métricas (loss, accuracy, tokens) en tensores residentes en GPU (`torch.tensor(..., device=device)`).
- **Sincronización Diferida:** La transferencia a CPU se realiza una única vez al finalizar el recorrido del dataset, reduciendo drásticamente la latencia de comunicación y permitiendo que la GPU procese colas de comandos sin interrupciones.

### 3. Flujo Asíncrono en Modelos Dinámicos (`neuro_architectures_v2.py`)
**Problema:** El modelo `NeuroModelV2` verificaba explícitamente si quedaban tokens activos (`if not active_mask.any():`) en cada capa. Esta operación es un "punto de sincronización" que obliga a la CPU a esperar a que la GPU termine de calcular la máscara actual antes de lanzar la siguiente capa.
**Solución:**
- **Ejecución Especulativa/Asíncrona:** Se eliminó la verificación bloqueante. PyTorch maneja eficientemente operaciones sobre tensores vacíos o enmascarados de forma asíncrona.

### 4. Throughput de Entrenamiento (`experiments/train_loop.py`)
**Problema:** El bucle de entrenamiento original medía el tiempo de cada paso individualmente y acumulaba la pérdida moviendo escalares a la CPU (`loss.item()`) en cada iteración.
**Solución:**
- **Medición por Epoch:** Se cambió la medición de tiempo para abarcar el epoch completo. Esto permite que el dataloader y la GPU superpongan trabajo (pipelining) de manera efectiva, ocultando la latencia de carga de datos y lanzamiento de kernels.
- **Acumulación Asíncrona:** La pérdida se acumula en un tensor de GPU y solo se sincroniza al final del epoch.

## Observaciones de Arquitectura

### GMA_MoE (Mixture of Experts)
La implementación actual en `models/gma_moe.py` ejecuta **todos** los expertos para cada token (`torch.stack([expert(hidden)...])`) y luego aplica una máscara suave.
- **Impacto:** Aunque funcionalmente correcto, esto no reduce el coste computacional (FLOPs) respecto a una red densa equivalente a la suma de expertos. Con 4 expertos, el coste es 4x el de un experto único, independientemente de la dispersión del router.
- **Nota:** Para obtener ventajas de velocidad real en GPU con MoE disperso, se requeriría una implementación con kernels personalizados (e.g., Triton/CUDA) o primitivas de `torch.scatter/gather` que eviten computar expertos inactivos. Dado el alcance actual (PyTorch puro), la implementación densa es aceptable para validación de calidad, pero las métricas de eficiencia deben interpretarse con esta salvedad.

## Estado de la Validación de Comparativas

El código ahora garantiza una medición más justa y precisa del rendimiento:
1. **Medición de Tiempo:** Los `DeviceTimer` (basados en `cuda.Event`) miden el tiempo de ejecución en GPU sin verse afectados por la latencia de Python/CPU que hemos optimizado.
2. **Métricas Reales:** La acumulación correcta de `RealCaseAccuracy` y `RealCaseLoss` asegura que los modelos se comparen por su capacidad real de generalización.
3. **Eficiencia de Memoria:** El cacheo en DCA reduce la fragmentación de memoria durante la inferencia.

## Recomendaciones para el Usuario

Para maximizar el rendimiento en su entorno Windows con GPU:
- **Workers de Datos:** Al ejecutar scripts, use `--num-workers 4` (o un valor cercano a sus núcleos físicos).
  ```bash
  python experiments/run_benchmark.py --num-workers 4
  ```
- **Precisión Mixta:** Asegúrese de usar `--amp-dtype bf16` si su GPU lo soporta (Ampere+), o `fp16` en caso contrario.
