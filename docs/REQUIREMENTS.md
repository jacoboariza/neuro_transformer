Contexto y Requisitos del Proyecto: neuro_transformer

1. Descripción General del Proyecto

El repositorio jacoboariza/neuro_transformer es un framework experimental reproducible diseñado para la investigación en Inteligencia Artificial. Su propósito principal es comparar de forma empírica el rendimiento y la eficiencia computacional de una arquitectura Transformer estándar y un modelo de lenguaje local real (HuggingFaceTB/SmolLM-135M) frente a una nueva familia de arquitecturas y módulos bio-inspirados fundamentados en los últimos avances de la neurociencia y la psicología cognitiva (2024-2026).

El objetivo último es demostrar empíricamente que la integración de principios biológicos (escasez neuronal, procesamiento predictivo, simulación contrafáctica, etc.) puede mitigar las ineficiencias críticas de los LLMs actuales (alto consumo energético, sobre-parametrización y olvido catastrófico).

2. Ecosistema de Arquitecturas a Evaluar

El framework debe soportar y comparar las siguientes topologías:

Baselines (Puntos de Control)

Transformer Estándar: Capa de atención densa clásica ($O(N^2)$).

Modelo SOTA Local: SmolLM-135M (Hugging Face) como ancla de rendimiento real.

Modelos Neuro-Inspirados Base (v1.0)

DCA (Dynamic Connectome Architecture): Enrutamiento escaso (sparse) imitando el conectoma biológico.

MOPN (Multi-dimensional Orthogonal Processing Networks): Proyecciones en subespacios latentes ortogonales para evitar interferencia de conceptos.

SCT (Sleep-Cycled Transformers): Consolidación metabólica y poda de pesos offline para aprendizaje continuo.

GMA-MoE (Glial Modulation & Mixture of Experts): Red glial paralela que asigna la carga de parámetros dinámicamente según la complejidad del input.

Módulos Cognitivos Avanzados (v2.0)

PMT (Predictive Minimalist Trace / Early Exit): Cortocircuito computacional basado en la predictibilidad de la información.

CEN (Counterfactual Episodic Network): Simulación latente de escenarios múltiples ("¿Y si...?") antes de la generación del token.

VLM (Vicarious Learning Module): Red estudiante diminuta que aprende en la sombra observando el espacio latente del modelo maestro.

3. Requisitos Técnicos Críticos y Restricciones (Checklist Activo)

Para garantizar la validez científica del benchmark y evitar artefactos de hardware o matemáticos, el código del framework debe cumplir estrictamente con los siguientes requisitos de implementación:

[ ] R1. Pipeline de Datos Reales: Queda prohibido el uso de tensores aleatorios (torch.randn) para medir la pérdida (Loss). El sistema debe utilizar un DataLoader alimentado por un dataset real (ej. HuggingFaceFW/fineweb-edu o wikitext) y usar el tokenizador oficial de SmolLM-135M (AutoTokenizer).

[ ] R2. Escasez (Sparsity) Computacional Real en DCA: La capa DCA no debe usar máscaras estáticas multiplicativas densas. Debe implementarse con tensores torch.sparse nativos o arquitectura Block-Sparse para asegurar que la reducción de FLOPs y el ahorro de VRAM sean genuinos a nivel de hardware.

[ ] R3. Early Exit a Nivel de Token (PMT): El mecanismo de salida temprana no puede usar sentencias return a nivel de secuencia o batch. Debe utilizar Enmascaramiento de Tokens (Token Masking): congelar los tokens con alta confianza y propagar únicamente los tokens con "sorpresa" a las capas más profundas de la red.

[ ] R4. Aislamiento de Gradientes en VLM: El submódulo de Aprendizaje Vicario debe estar matemáticamente aislado de la red principal. Se requiere el uso estricto de .detach() en los estados ocultos antes de introducirlos en la red estudiante para evitar la contaminación de la retropropagación (backpropagation) del maestro.

[ ] R5. Métricas de Rendimiento Precisas (Profiling): Los tiempos de inferencia y entrenamiento deben medirse utilizando eventos nativos de hardware cuando se ejecute en GPU (ej. torch.cuda.Event(enable_timing=True)), excluyendo los tiempos de transferencia de memoria asíncrona de los resultados.

4. Resultado Esperado

Al ejecutar el benchmark, el sistema deberá generar un reporte (CSV y visualizaciones) que evidencie el trade-off entre Parámetros Entrenables, FLOPs / Tiempo de Inferencia y Perplejidad Final (Loss), validando qué combinaciones bio-inspiradas ofrecen la mejor relación precisión-eficiencia.