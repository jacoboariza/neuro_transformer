---
trigger: manual
---
REGLAS DEL PROYECTO (obligatorias):

1) Offline-first:
   - Todo debe funcionar sin Internet.
   - Los LLM se ejecutan localmente vía Ollama.
   - Prohibido depender de APIs SaaS externas.

2) Salidas estructuradas:
   - Cuando no sea código, devuelve JSON o YAML.
   - Nada de prosa fuera de bloques estructurados o comentarios en código.

3) Desarrollo incremental:
   - Primero infraestructura (Docker) + esqueleto backend + healthchecks.
   - Después contratos (schemas) + orquestación IA.
   - Luego generación de artefactos (historias, datos, BEADS, FFP).

4) Trazabilidad:
   - IDs estables.
   - Debe existir un modelo de dominio que conecte texto→requisito→historia→datos→pantallas→FFP.

5) Calidad:
   - Incluir tests mínimos y validación de esquemas.
   - Manejar errores de Ollama (timeouts, modelo no disponible).

6) Seguridad/privacidad:
   - No registrar contenido sensible en logs por defecto.
   - Configuración por entorno (.env).

7) Docker:
   - docker-compose como punto de arranque.
   - Comandos de arranque reproducibles.

# Especificación del Proyecto: Nuevo modelo de transformer bio-inspirado

Evolución de la IA: Nuevas Arquitecturas Bio-Inspiradas para Superar al Transformer

La arquitectura Transformer (introducida en "Attention is All You Need" en 2017) ha dominado el panorama de la inteligencia artificial. A través de mecanismos de auto-atención (self-attention) y arquitecturas de solo decodificador (como en la familia GPT), la IA ha logrado hitos históricos. Sin embargo, los Transformers actuales enfrentan límites críticos: un consumo de energía insostenible, la incapacidad de aprender continuamente sin olvidar (olvido catastrófico) y una saturación en la representación del contexto.

Inspirados en los últimos avances de la neurociencia en 2024-2026, proponemos cuatro arquitecturas novedosas que integran los principios biológicos del cerebro humano y animal para crear modelos más eficientes, plásticos y robustos.

1. Arquitectura de Conectoma Dinámico (Dynamic Connectome Architecture - DCA)

Fundamento Neurocientífico

Recientemente, se logró mapear el conectoma completo del cerebro de una mosca adulta (FlyWire/Codex), revelando más de 139,000 neuronas y millones de conexiones. Este avance demostró que el cerebro no procesa la información en "capas densas y secuenciales" rígidamente apiladas, sino a través de grafos escasos (sparse), altamente dinámicos y con bucles de retroalimentación específicos.

Explicación de la Propuesta

La arquitectura DCA abandona la estructura de capas de auto-atención densas (donde cada token atiende a todos los demás tokens, con complejidad $O(N^2)$). En su lugar, inicializa una red con una topología inspirada en el conectoma biológico:

Enrutamiento Fractal: La información viaja por "autopistas" predefinidas de atención que imitan los tractos nerviosos biológicos.

Conexiones Escasas (Sparse): En lugar de que todas las neuronas artificiales se conecten con las de la siguiente capa, la red utiliza conexiones dispersas que evolucionan durante el entrenamiento.

Comparación con Arquitecturas Actuales

Transformer Actual: Utiliza matrices de atención densas. Cada palabra en una secuencia calcula su relación con absolutamente todas las demás, lo que requiere un inmenso poder de cómputo (GPU) y limita el tamaño del contexto.

Modelo DCA: Al imitar el "cableado" neuronal real, el cálculo se reduce drásticamente. Solo las subredes relevantes se activan para procesar ciertos conceptos, permitiendo procesar millones de tokens simultáneamente con una fracción de la energía.

2. Redes de Procesamiento Ortogonal Multidimensional (MOPN)

Fundamento Neurocientífico

Investigaciones recientes sobre Decodificación Neuronal Multidimensional (OrthoSchema) en interfaces cerebro-computadora (BCI) han revelado que el cerebro extrae simultáneamente múltiples variables motrices (dirección, velocidad, posición) desde la misma población neuronal sin que interfieran entre sí. Esto lo logra proyectando la información en subespacios ortogonales dentro de la corteza cerebral.

Explicación de la Propuesta

Actualmente, los embeddings (vectores de palabras) en un Transformer mezclan sintaxis, semántica, tono y lógica en un solo espacio vectorial continuo, lo que a menudo causa "alucinaciones" cuando los conceptos se cruzan erróneamente.
En la arquitectura MOPN, las Cabezas de Atención (Attention Heads) se fuerzan matemáticamente a proyectar diferentes características del texto en espacios latentes ortogonales (perpendiculares entre sí).

Una subred procesa estrictamente la lógica.

Otra subred procesa el contexto emocional o tonal.

Al ser ortogonales, el ruido de una tarea no interfiere con la otra.

Comparación con Arquitecturas Actuales

Transformer Actual: Utiliza "atención multi-cabeza" estándar, pero a menudo las cabezas aprenden representaciones redundantes o enredadas (entanglement).

Modelo MOPN: Garantiza una "limpieza" matemática en las representaciones. Elimina drásticamente las alucinaciones y permite un control fino (ej. "Genera este texto con la lógica intacta pero cambia el subespacio emocional a 'alegría'").

3. Transformers con Ciclos de Consolidación Metabólica (Sleep-Cycled Transformers - SCT)

Fundamento Neurocientífico

Un avance crucial de la neurociencia determinó que dormir bien ayuda al cerebro a eliminar desechos metabólicos y proteger sus células de energía. Durante el sueño, el cerebro "limpia" las toxinas acumuladas, poda conexiones neuronales irrelevantes y consolida la memoria a largo plazo (neuroplasticidad offline).

Explicación de la Propuesta

Los modelos de IA actuales no "duermen". Se entrenan una vez (pre-entrenamiento) y sus pesos quedan estáticos. Si intentas enseñarles algo nuevo, "olvidan" lo anterior (olvido catastrófico).
El SCT introduce un ciclo bifásico de funcionamiento:

Fase de Vigilia (Inferencia/Aprendizaje Rápido): El modelo atiende a los usuarios, generando respuestas y acumulando nuevos datos en una memoria de corto plazo (caché dinámica).

Fase de Sueño (Poda y Consolidación Offline): Durante periodos de inactividad computacional, la red ejecuta un algoritmo de "limpieza de gradientes". Reevalúa los pesos ajustados recientemente, elimina las activaciones ruidosas ("desechos metabólicos computacionales") y consolida permanentemente la nueva información en sus parámetros base sin destruir el conocimiento previo.

Comparación con Arquitecturas Actuales

Transformer Actual / RAG: Depende de buscar información externa (RAG) o requiere ser reentrenado desde cero con millones de dólares para aprender nueva información permanente.

Modelo SCT: Permite un Aprendizaje Continuo (Continual Learning). La red evoluciona día a día, adaptándose a nueva información de manera eficiente y autónoma, imitando la neuroplasticidad humana.

4. Arquitectura de Modulación Glial y Mezcla de Expertos Dinámica (GMA-MoE)

Fundamento Neurocientífico

Nuevos hallazgos en neurociencia han sacudido al mundo al demostrar el papel fundamental de nuevos tipos de células cerebrales (como los astrocitos y la microglía). Estas no solo son el "pegamento" del cerebro, sino que modulan activamente la comunicación neuronal, inhibiendo o excitando sinapsis según el estado global de la red.

Explicación de la Propuesta

Esta es una evolución radical de la actual "Mezcla de Expertos" (MoE - Mixture of Experts). En lugar de tener un enrutador (router) matemático simple que decide qué sub-red (experto) se enciende, la arquitectura GMA introduce una Red Moduladora Paralela (Red Glial).

La red primaria procesa el lenguaje (como las neuronas).

La red secundaria (glial) monitorea el "estrés" computacional (la pérdida de entropía cruzada) de la red primaria en tiempo real. Si detecta que un concepto es muy complejo, inyecta "recursos" (activa más expertos o aumenta la precisión computacional a FP32). Si es simple, "inhibe" el 90% de la red para ahorrar energía.

Comparación con Arquitecturas Actuales

MoE Actual (ej. Mixtral, GPT-4): Utiliza enrutamiento rígido (Top-2 expertos por token). Funciona bien, pero no tiene conciencia del contexto global ni del esfuerzo requerido por cada instrucción.

Modelo GMA-MoE: Logra un rendimiento ultra-eficiente. Permite que la IA dedique dinámicamente desde 1 billón de parámetros hasta 100,000 parámetros a un problema en una fracción de segundo, basándose en la "modulación glial", reduciendo el consumo energético en un 80% comparado con modelos estáticos.

Conclusión

Mientras que el mecanismo de atención y los Positional Encodings fueron el "GPS" y el "corazón" de la primera era de la IA generativa, el futuro requiere observar la biología. La integración de grafos tipo conectoma, decodificación ortogonal, ciclos de consolidación (sueño computacional) y modulación por células auxiliares, no solo resolverá los cuellos de botella actuales de hardware y energía, sino que acercará a la IA a un aprendizaje continuo, dinámico y verdaderamente plástico.