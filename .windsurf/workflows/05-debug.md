---
title: debug
description: Flujo sistemático de depuración- reproducir → analizar → corregir → validar
---

## Paso 0 — Descripción del problema
- Describe el error de forma precisa:
  - Qué ocurre
  - Qué debería ocurrir
  - Desde cuándo sucede
- Indica entorno(s) afectados: local, docker, staging, prod.
- Adjunta mensajes de error, logs o trazas si existen.

## Paso 1 — Reproducción controlada
- Define los pasos mínimos para reproducir el problema.
- Confirma si el error es:
  - Reproducible siempre
  - Intermitente
  - Dependiente de datos o estado
- Si NO se puede reproducir, DETENTE y documenta hipótesis de por qué.

## Paso 2 — Aislamiento del problema
- Reduce el ámbito:
  - Módulo
  - Función
  - Endpoint
  - Configuración
- Desactiva componentes no esenciales si es posible.
- Añade logs temporales SOLO para aislar el fallo.

## Paso 3 — Formulación de hipótesis
- Enumera 2–5 hipótesis plausibles.
- Para cada hipótesis, indica:
  - Qué la apoyaría
  - Qué la descartaría
- Prioriza por probabilidad e impacto.

## Paso 4 — Verificación de hipótesis
- Diseña pequeñas pruebas para confirmar o descartar hipótesis.
- Cambia UNA cosa cada vez.
- Documenta el resultado de cada prueba.

## Paso 5 — Identificación de la causa raíz
- Declara explícitamente la causa raíz confirmada.
- Indica:
  - Por qué ocurre
  - Por qué no se detectó antes
  - Qué lo dispara

## Paso 6 — Diseño de la corrección
- Define el cambio mínimo necesario para corregir el problema.
- Evalúa efectos colaterales y regresiones posibles.
- Decide si hace falta refactor o solo ajuste puntual.

## Paso 7 — Implementación del fix
- Aplica el cambio de forma clara y localizada.
- Evita “aprovechar” para meter mejoras no relacionadas.
- Mantén el fix pequeño y revisable.

## Paso 8 — Test de regresión
- Añade o actualiza tests que fallen sin el fix y pasen con él.
- Si no hay framework de tests, crea una verificación manual documentada.
- Asegura que el test cubre la causa raíz, no solo el síntoma.

## Paso 9 — Validación completa
- Reproduce de nuevo el escenario original.
- Ejecuta tests existentes relevantes.
- Elimina logs temporales añadidos durante el debug.

## Paso 10 — Cierre y aprendizaje
- Resume en 3–5 bullets:
  - Causa raíz
  - Solución aplicada
  - Cómo prevenirlo en el futuro
- Si procede, crea o actualiza una memory con la lección aprendida.

## Checklist final (obligatorio)
- [ ] Problema reproducido y documentado
- [ ] Causa raíz identificada
- [ ] Fix mínimo aplicado
- [ ] Test de regresión añadido
- [ ] Sin efectos colaterales conocidos

## Salida esperada
- Bug corregido
- Tests cubriendo el fallo
- Conocimiento capturado para evitar recurrencia