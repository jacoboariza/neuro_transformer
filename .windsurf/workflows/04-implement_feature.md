---
title: implement_feature
description: Convierte un plan aprobado en una feature completa lista para PR
---

## Paso 0 — Contexto y alcance
- Identifica claramente la feature a implementar.
- Enumera los requisitos funcionales y no funcionales afectados.
- Confirma qué partes del sistema quedan fuera de alcance.

## Paso 1 — Validación del plan
- Relee el plan técnico aprobado.
- Verifica dependencias, riesgos y supuestos.
- Si falta información o hay ambigüedad, DETENTE y formula preguntas concretas antes de tocar código.

## Paso 2 — Preparación del entorno
- Verifica que el entorno local funciona con `docker compose up`.
- Comprueba versiones de dependencias relevantes.
- Asegura que los tests existentes pasan antes de cambiar nada.

## Paso 3 — Diseño fino de la implementación
- Divide la feature en tareas pequeñas e incrementales.
- Para cada tarea, define:
  - Qué archivo(s) se modifican
  - Qué nueva lógica se introduce
  - Qué test valida el cambio
- Ordena las tareas para minimizar riesgo.

## Paso 4 — Implementación incremental
- Implementa una tarea cada vez.
- Después de cada cambio:
  - Ejecuta tests relevantes
  - Verifica logs y comportamiento esperado
- No acumules cambios no verificados.

## Paso 5 — Tests y validación
- Añade tests nuevos donde haya lógica nueva.
- Ajusta tests existentes si el comportamiento cambia.
- Asegura cobertura mínima razonable para la feature.

## Paso 6 — Limpieza y calidad
- Elimina código muerto, comentarios obsoletos y logs innecesarios.
- Aplica formateo y linting del proyecto.
- Revisa nombres de variables, funciones y ficheros.

## Paso 7 — Documentación mínima
- Actualiza README o documentación técnica si aplica.
- Documenta decisiones relevantes tomadas durante la implementación.
- Añade notas de uso o ejemplos si la feature lo requiere.

## Paso 8 — Preparación del PR
- Resume los cambios en bullets claros.
- Indica:
  - Qué problema resuelve
  - Cómo se ha probado
  - Riesgos o impactos conocidos
- Verifica que el PR sea pequeño, entendible y revisable.

## Paso 9 — Checklist final (obligatorio)
- [ ] Entorno levanta con Docker
- [ ] Tests pasan
- [ ] No hay secretos ni datos sensibles
- [ ] Código alineado con las rules del proyecto
- [ ] Feature lista para revisión

## Salida esperada
- Feature completamente implementada
- Tests pasando
- Documentación mínima actualizada
- Pull Request listo para revisión
