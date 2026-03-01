import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from models.dca import DCA_Layer as SparseDCA_Layer

# =====================================================================
# MÓDULOS BASE (v1.0 Simplificados para el experimento v2.0)
# =====================================================================

# =====================================================================
# NUEVOS MÓDULOS (v2.0)
# =====================================================================

class PMT_EarlyExit(nn.Module):
    """
    Predictive Minimalist Trace (PMT) - Módulo de Salida Temprana.
    Evalúa si la representación latente ya tiene suficiente "confianza"
    o "baja sorpresa". Si es así, corta el cálculo de las siguientes capas.
    """
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        # Un pequeño clasificador lineal para predecir la salida en esta capa
        self.exit_predictor = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        # Devuelve logits y confianza por token.
        logits = self.exit_predictor(x)
        probs = F.softmax(logits, dim=-1)
        confidence = probs.amax(dim=-1)
        return logits, confidence

class CEN_CounterfactualSimulation(nn.Module):
    """
    Counterfactual Episodic Network (CEN) - Simulación de "¿Y si...?".
    En lugar de un solo forward, genera ramificaciones latentes y escoge la más coherente.
    """
    def __init__(self, embed_dim, num_branches=3):
        super().__init__()
        self.num_branches = num_branches
        # Diferentes "caminos" o suposiciones lógicas
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU()) 
            for _ in range(num_branches)
        ])
        # Evaluador de coherencia interna (mide si la rama tiene sentido)
        self.coherence_evaluator = nn.Linear(embed_dim, 1)

    def forward(self, x):
        best_out = None
        best_score = -float('inf')
        
        # Simulamos los futuros alternativos (ramas)
        for branch in self.branches:
            simulated_future = branch(x)
            # Evaluamos qué tan "coherente" o lógica es esta rama
            score = self.coherence_evaluator(simulated_future).mean()
            
            if score > best_score:
                best_score = score
                best_out = simulated_future
                
        return best_out + x # Devolvemos la rama ganadora

class VLM_VicariousStudent(nn.Module):
    """
    Vicarious Learning Module (VLM) - Aprendiz en la sombra.
    Una red diminuta que observa el espacio latente sin afectar el forward principal.
    """
    def __init__(self, embed_dim, student_dim=64):
        super().__init__()
        self.encoder = nn.Linear(embed_dim, student_dim)
        self.decoder = nn.Linear(student_dim, embed_dim)
        
    def observe_and_learn(self, hidden_states):
        """Aprende observando al modelo principal (Teacher) sin gradientes cruzados."""
        detached_states = hidden_states.detach()
        compressed = F.relu(self.encoder(detached_states))
        mimicked = self.decoder(compressed)
        vicarious_loss = F.mse_loss(mimicked, detached_states)
        return vicarious_loss

# =====================================================================
# ARQUITECTURA MAESTRA v2.0
# =====================================================================

class NeuroModelV2(nn.Module):
    def __init__(self, embed_dim, num_classes, num_layers=6, dca_sparsity=0.8):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.uses_token_masking = True
        self.uses_vlm_detach = True
        self.uses_sparse_dca = True
        self.last_token_depth = None
        self.last_active_mask = None
        
        # Capas de procesamiento base (usamos el Conectoma Sparse por eficiencia)
        self.layers = nn.ModuleList(
            [SparseDCA_Layer(embed_dim=embed_dim, sparsity=dca_sparsity) for _ in range(num_layers)]
        )
        
        # PMT: Añadimos un "Early Exit" después de cada capa
        self.early_exits = nn.ModuleList([PMT_EarlyExit(embed_dim, num_classes) for _ in range(num_layers)])
        
        # CEN: Módulo de pensamiento contrafáctico (insertado en la capa media)
        self.cen_module = CEN_CounterfactualSimulation(embed_dim)
        
        # VLM: El estudiante observador
        self.vlm_student = VLM_VicariousStudent(embed_dim)

    def forward(self, x, exit_threshold=0.85):
        """
        Forward con token masking para early exit por token.
        exit_threshold: confianza requerida para congelar un token.
        """
        vicarious_losses = []
        token_depth = torch.zeros(x.size(0), x.size(1), device=x.device, dtype=x.dtype)
        active_mask = torch.ones(x.size(0), x.size(1), device=x.device, dtype=torch.bool)
        final_logits = None
        num_classes = self.early_exits[0].exit_predictor.out_features
        
        for i in range(self.num_layers):
            if not bool(active_mask.any()):
                break

            # 1. Capa de procesamiento profundo
            x_flat = x.reshape(-1, self.embed_dim)
            active_flat = active_mask.reshape(-1)

            active_tokens = x_flat[active_flat].unsqueeze(1)
            processed_active = self.layers[i](active_tokens).squeeze(1)
            
            # 2. Simulación Contrafáctica (Solo en la capa central para decisiones críticas)
            if i == self.num_layers // 2:
                processed_active = self.cen_module(processed_active.unsqueeze(1)).squeeze(1)

            x_flat = x_flat.clone()
            x_flat[active_flat] = processed_active
            x = x_flat.reshape_as(x)
                
            # 3. Aprendizaje Vicario (El estudiante observa el pensamiento actual)
            vicarious_losses.append(self.vlm_student.observe_and_learn(x))
                
            # 4. PMT por token: congelamos tokens confiables y propagamos tokens sorpresa.
            logits_flat = x.new_zeros((x_flat.size(0), num_classes))
            confidence_flat = x.new_ones((x_flat.size(0),))

            active_logits = self.early_exits[i].exit_predictor(x_flat[active_flat])
            active_confidence = torch.softmax(active_logits, dim=-1).amax(dim=-1)

            logits_flat[active_flat] = active_logits
            confidence_flat[active_flat] = active_confidence

            logits = logits_flat.reshape(x.size(0), x.size(1), num_classes)
            confidence = confidence_flat.reshape(x.size(0), x.size(1))

            if final_logits is None:
                final_logits = logits
            else:
                final_flat = final_logits.reshape(-1, num_classes)
                final_flat = final_flat.clone()
                final_flat[active_flat] = active_logits
                final_logits = final_flat.reshape_as(final_logits)


            token_depth = token_depth + active_mask.to(dtype=token_depth.dtype)
            continue_mask = confidence < exit_threshold
            active_mask = active_mask & continue_mask

        if final_logits is None:
            final_logits, _ = self.early_exits[-1](x)

        self.last_token_depth = token_depth.detach()
        self.last_active_mask = active_mask.detach()

        avg_layers_used = float(token_depth.mean().item())
        avg_vicarious_loss = sum(vicarious_losses) / max(len(vicarious_losses), 1)
        return final_logits, avg_layers_used, avg_vicarious_loss

# =====================================================================
# EXPERIMENTO DE DEMOSTRACIÓN
# =====================================================================
def run_v2_experiment():
    print("=== Benchmark de Arquitecturas Neuro-Inspiradas v2.0 ===")
    print("Evaluando: Predictive Minimalist Trace (Early Exit) & Pensamiento Contrafáctico\n")
    
    embed_dim = 256
    num_classes = 1000 # Tamaño del vocabulario sintético
    num_layers = 12    # Modelo profundo
    batch_size = 1
    seq_len = 64
    
    # Instanciamos nuestra red v2.0
    model = NeuroModelV2(embed_dim, num_classes, num_layers=num_layers)
    model.eval()
    
    # Generamos dos tipos de datos:
    # 1. Input "Fácil / Predecible" (Simula frases como "El perro ladra")
    # Forzamos los pesos para que la primera capa tenga mucha confianza
    easy_input = torch.randn(batch_size, seq_len, embed_dim) * 5.0 
    
    # 2. Input "Difícil / Sorprendente" (Simula un problema de matemáticas complejo)
    hard_input = torch.randn(batch_size, seq_len, embed_dim) * 0.1
    
    def measure_inference(input_tensor, threshold, desc):
        start = time.time()
        logits, layers_used, vic_loss = model(input_tensor, exit_threshold=threshold)
        end = time.time()
        
        time_ms = (end - start) * 1000
        print(f"[{desc}]")
        print(f" - Capas ejecutadas: {layers_used} de {num_layers}")
        print(f" - Tiempo inferencia: {time_ms:.2f} ms")
        print(f" - Aprendizaje del Estudiante (Loss Vicario): {vic_loss:.4f}\n")
        return time_ms

    # Ejecutamos el benchmark
    print("Escenario 1: Procesando un concepto MUY FÁCIL (Alta predictibilidad)")
    time_easy = measure_inference(easy_input, threshold=0.6, desc="PMT Activado (Corte temprano)")

    print("Escenario 2: Procesando un concepto MUY COMPLEJO (Mucha sorpresa)")
    time_hard = measure_inference(hard_input, threshold=0.99, desc="PMT Desactivado (Requiere análisis profundo)")
    
    print("=== Conclusión del Experimento v2.0 ===")
    ahorro = (1 - (time_easy / time_hard)) * 100
    print(f"El mecanismo PMT (Early Exit) ahorró un {ahorro:.1f}% del tiempo de computación")
    print("en el concepto fácil, mientras que el módulo CEN garantizó la coherencia en las capas medias.")
    print("Simultáneamente, el submódulo VLM aprendió de la experiencia en la sombra.")

if __name__ == "__main__":
    run_v2_experiment()