---
title: req2design
description: Convierte requisitos funcionales en diseño funcional y técnico listo para implementar
---

## Paso 0 — Fuente de requisitos
- Identifica el documento fuente de requisitos (ej. docs/requisitos_funcionales.md).
- Confirma versión y fecha del documento.
- No asumas requisitos no escritos explícitamente.

## Paso 1 — Comprensión funcional
- Resume los requisitos en:
  - Objetivo del sistema
  - Actores implicados
  - Flujos principales
- Detecta ambigüedades o lagunas.
- Si falta información crítica, DETENTE y formula preguntas concretas.

## Paso 2 — Alcance y límites
- Define claramente:
  - Qué hace el sistema
  - Qué NO hace el sistema
- Identifica dependencias externas (si existen).
- Señala supuestos explícitos.

## Paso 3 — Diseño funcional
- Describe los flujos funcionales principales paso a paso.
- Para cada flujo indica:
  - Entrada
  - Proceso
  - Salida
- Usa lenguaje funcional, no técnico todavía.

## Paso 4 — Diseño lógico / técnico
- Define componentes principales (módulos, servicios, capas).
- Describe responsabilidades de cada componente.
- Identifica interfaces entre componentes (inputs / outputs).
- Indica dónde interviene el LLM y con qué propósito.

## Paso 5 — Modelo de datos (alto nivel)
- Enumera entidades principales.
- Describe atributos clave.
- Relaciona entidades cuando aplique.
- No bajes aún a SQL o schemas concretos.

## Paso 6 — Estrategia de IA
- Define:
  - Qué tareas hace el LLM
  - Qué NO debe hacer el LLM
  - Qué entradas recibe
  - Qué salidas produce
- Indica si la interacción es síncrona o asíncrona.
- Confirma uso de LLM local vía Ollama.

## Paso 7 — Errores y casos límite
- Identifica errores esperables:
  - Entrada inválida
  - Respuestas incompletas del LLM
  - Timeouts o fallos de infraestructura
- Define comportamiento esperado ante cada caso.

## Paso 8 — Criterios de aceptación
- Define criterios verificables para:
  - Cada flujo funcional
  - Cada componente clave
- Deben poder validarse con tests o verificaciones manuales.

## Paso 9 — Artefactos de salida
- Genera:
  - Diseño funcional estructurado
  - Diseño técnico de alto nivel
  - Lista de decisiones tomadas
- Indica qué partes están listas para `/implement_feature`.

## Checklist final (obligatorio)
- [ ] Requisitos comprendidos y sin ambigüedades críticas
- [ ] Alcance claramente definido
- [ ] Diseño funcional completo
- [ ] Diseño técnico coherente
- [ ] Estrategia de IA definida
- [ ] Listo para implementación

## Salida esperada
- Documento de diseño funcional y técnico
- Base sólida para implementar sin improvisar
