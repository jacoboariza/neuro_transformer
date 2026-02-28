# Reglas base del proyecto (siempre)

- No asumas requisitos: si falta info, pregunta primero.
- Antes de tocar código: explica el plan en 5-10 bullets.
- Cambios pequeños e incrementales: un commit lógico por paso.
- Seguridad: nunca metas secretos en código ni en logs.
- Testing: si tocas lógica, añade/actualiza tests.
- Docker: todo lo ejecutable debe correr con docker compose.
- Observabilidad: logs útiles, sin PII, y con niveles.
- Si usas algún modelo de IA dale preferencia a usarlo con ollama en local, para mejorar el rendimiento.
