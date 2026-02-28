# Reglas para infraestructura y Docker

- No rompas compatibilidad con Windows.
- Prefiere imágenes slim y multi-stage cuando aplique.
- No uses latest en imágenes salvo herramientas de dev.
- Añade healthchecks si hay servicios.
