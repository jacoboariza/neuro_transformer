import torch

if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    print(f"✅ Dispositivo reconocido: {torch.cuda.get_device_name(0)}")
    print(f"🚀 Capacidad de cómputo: sm_{major}{minor}")
    
    # Prueba de cálculo real en la GPU
    x = torch.randn(2048, 2048, device='cuda')
    y = torch.matmul(x, x)
    print("💎 ¡Cálculo en arquitectura Blackwell completado con éxito!")
else:
    print("❌ Error: CUDA sigue sin ser detectado correctamente.")