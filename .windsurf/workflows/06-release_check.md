---
title: release_check
description: Checklist de release - versionado, changelog y smoke tests antes de merge o deploy
---

## Paso 0 — Contexto del release
- Indica:
  - Tipo de release: patch / minor / major
  - Rama origen y rama destino
  - Entorno objetivo: staging / prod
- Enumera PRs o features incluidas.

## Paso 1 — Versionado
- Determina el incremento de versión según impacto:
  - Patch: bugfix sin cambio de comportamiento
  - Minor: nueva funcionalidad compatible
  - Major: breaking change
- Actualiza la versión en:
  - Archivo(s) de versión del proyecto
  - Imágenes Docker (tags)
- Verifica coherencia entre código, Docker y documentación.

## Paso 2 — Changelog
- Actualiza el CHANGELOG con:
  - Nuevas funcionalidades
  - Bugfixes
  - Cambios potencialmente rompientes
- Usa lenguaje claro y orientado a usuario.
- Incluye referencias a PRs o issues si aplica.

## Paso 3 — Revisión de configuración
- Verifica variables de entorno:
  - Obligatorias
  - Nuevas
  - Deprecadas
- Confirma valores por defecto seguros.
- Asegura compatibilidad con entornos existentes.

## Paso 4 — Smoke tests
- Define el conjunto mínimo de pruebas críticas:
  - Arranque del sistema
  - Endpoints clave / flujos principales
  - Integraciones externas críticas
- Ejecuta smoke tests en entorno limpio con Docker.
- Documenta comandos exactos y resultados.

## Paso 5 — Validación técnica
- Confirma:
  - `docker compose up` funciona sin errores
  - No hay warnings críticos en logs
  - Tests automáticos pasan
- Revisa consumo anómalo de recursos si es relevante.

## Paso 6 — Seguridad y cumplimiento
- Verifica:
  - No hay secretos en código ni imágenes
  - Dependencias actualizadas sin CVEs críticos conocidos
  - Permisos mínimos necesarios
- Confirma alineación con rules del proyecto.

## Paso 7 — Comunicación del release
- Prepara resumen del release:
  - Qué se entrega
  - Riesgos conocidos
  - Pasos de rollback
- Define responsables y ventana de despliegue si aplica.

## Paso 8 — Aprobación final
- Revisa checklist completo.
- Confirma que el release es:
  - Reproducible
  - Reversible
  - Entendible para terceros
- Autoriza merge o deploy.

## Checklist final (obligatorio)
- [ ] Versión incrementada correctamente
- [ ] Changelog actualizado
- [ ] Smoke tests ejecutados y documentados
- [ ] Docker y configuración validados
- [ ] Sin riesgos críticos conocidos

## Salida esperada
- Release validado
- Información clara para despliegue
- Riesgos controlados
