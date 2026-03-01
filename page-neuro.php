<?php
/* Template Name: Neuro Transformer SPA */
?>

<!DOCTYPE html>
<html lang="es" class="scroll-smooth">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neuro Transformer | Evolución Bio-Inspirada de la IA</title>
    
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <!-- Tailwind Configuration & Custom Styles -->
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        primary: '#1e3a8a',    // Blue 900
                        secondary: '#3b82f6',  // Blue 500
                        accent: '#10b981',     // Emerald 500
                        background: '#f8fafc', // Slate 50
                        surface: '#ffffff',
                    },
                    fontFamily: {
                        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
                    }
                }
            }
        }
    </script>
    
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        body {
            font-family: 'Inter', sans-serif;
            background-color: #f8fafc;
            color: #1e293b;
        }

        /* CRITICAL: Chart Container Styling for responsiveness and bounds */
        .chart-container {
            position: relative;
            width: 100%;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
            height: 350px;
            max-height: 400px;
        }

        @media (min-width: 768px) {
            .chart-container {
                height: 400px;
            }
        }

        /* Custom utilities */
        .glass-card {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(226, 232, 240, 0.8);
        }
        
        .tab-active {
            border-bottom: 2px solid #3b82f6;
            color: #1e3a8a;
            font-weight: 600;
        }
        
        .tab-inactive {
            color: #64748b;
        }
        
        .tab-inactive:hover {
            color: #3b82f6;
        }

        /* Hide scrollbar for clean horizontal scrolling in tabs if needed */
        .no-scrollbar::-webkit-scrollbar {
            display: none;
        }
        .no-scrollbar {
            -ms-overflow-style: none;
            scrollbar-width: none;
        }
    </style>
</head>
<body class="antialiased selection:bg-secondary selection:text-white relative">

    <!-- REQUIRED PLACEHOLDER COMMENTS -->
    <!-- Chosen Palette: Slate Backgrounds, Deep Blue Primary, Vibrant Blue Secondary, Emerald Accents for a professional, clinical, yet modern tech feel. -->
    <!-- Application Structure Plan: The SPA is designed as a linear narrative dashboard. It starts with the Hero (hook), flows into the Problem (context), presents the Solutions (tabbed interface for v1.0 and v2.0 to manage cognitive load), verifies Methodology (interactive checklist), and concludes with Results (interactive charts). This structure guides the user from understanding the 'why' to exploring the 'how' and proving the 'what', making complex neuro-AI concepts digestible. -->
    <!-- Visualization & Content Choices: 
         1. Problem Section -> Goal: Inform & Compare -> Viz: Grid of cards with Unicode icons -> Interaction: Hover effects to reveal depth. -> Library: Tailwind.
         2. Architectures (v1 & v2) -> Goal: Organize & Detail -> Viz: Tabbed interface with clickable detail cards -> Interaction: Click tabs to switch versions, click cards to expand info without leaving the context. -> Library: Vanilla JS + Tailwind.
         3. Methodology -> Goal: Inform -> Viz: Interactive Checklist -> Interaction: Clickable items that fill a progress bar, creating engagement. -> Library: Vanilla JS.
         4. Results -> Goal: Compare Relationships -> Viz: Bar Chart (Time) and Scatter Plot (Trade-off) -> Interaction: Hover tooltips for exact metrics. -> Library: Chart.js (NO SVG used).
         CONFIRMATION: NO SVG graphics used. NO Mermaid JS used. -->

    <!-- Navigation -->
    <nav class="fixed w-full z-50 glass-card transition-all duration-300" id="navbar">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="flex justify-between items-center h-16">
                <div class="flex-shrink-0 flex items-center cursor-pointer" onclick="window.scrollTo(0,0)">
                    <span class="text-2xl mr-2">🧠</span>
                    <span class="font-bold text-xl text-primary tracking-tight">Neuro Transformer</span>
                </div>
                <div class="hidden md:flex space-x-8">
                    <a href="#problema" class="text-slate-600 hover:text-secondary transition-colors font-medium">El Problema</a>
                    <a href="#arquitecturas" class="text-slate-600 hover:text-secondary transition-colors font-medium">Arquitecturas</a>
                    <a href="#metodologia" class="text-slate-600 hover:text-secondary transition-colors font-medium">Metodología</a>
                    <a href="#resultados" class="text-slate-600 hover:text-secondary transition-colors font-medium">Benchmark</a>
                </div>
                <div class="hidden md:flex">
                    <a href="#resultados" class="bg-primary hover:bg-blue-800 text-white px-4 py-2 rounded-lg font-medium transition-colors shadow-md">
                        Ver Resultados
                    </a>
                </div>
                <!-- Mobile menu button -->
                <div class="md:hidden flex items-center">
                    <button id="mobile-menu-btn" class="text-slate-600 hover:text-primary focus:outline-none text-2xl">
                        ☰
                    </button>
                </div>
            </div>
        </div>
        <!-- Mobile Menu -->
        <div id="mobile-menu" class="hidden md:hidden bg-white border-t border-slate-200">
            <div class="px-2 pt-2 pb-3 space-y-1 sm:px-3">
                <a href="#problema" class="block px-3 py-2 text-slate-600 hover:bg-slate-50 font-medium rounded-md">El Problema</a>
                <a href="#arquitecturas" class="block px-3 py-2 text-slate-600 hover:bg-slate-50 font-medium rounded-md">Arquitecturas</a>
                <a href="#metodologia" class="block px-3 py-2 text-slate-600 hover:bg-slate-50 font-medium rounded-md">Metodología</a>
                <a href="#resultados" class="block px-3 py-2 text-slate-600 hover:bg-slate-50 font-medium rounded-md">Benchmark</a>
            </div>
        </div>
    </nav>

    <!-- 1. Hero Section -->
    <section class="relative pt-32 pb-20 lg:pt-40 lg:pb-28 overflow-hidden">
        <!-- Background decorative elements -->
        <div class="absolute top-0 left-0 w-full h-full overflow-hidden -z-10">
            <div class="absolute -top-[10%] -right-[10%] w-[50%] h-[50%] rounded-full bg-blue-100 blur-[100px] opacity-60"></div>
            <div class="absolute top-[40%] -left-[10%] w-[40%] h-[40%] rounded-full bg-emerald-50 blur-[100px] opacity-60"></div>
        </div>

        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <span class="inline-block py-1 px-3 rounded-full bg-blue-100 text-blue-800 text-sm font-semibold tracking-wider mb-6 border border-blue-200 shadow-sm">
                INVESTIGACIÓN DE CÓDIGO ABIERTO
            </span>
            <h1 class="text-4xl md:text-5xl lg:text-7xl font-extrabold text-primary tracking-tight leading-tight mb-6">
                El Futuro de la IA <br class="hidden md:block">
                <span class="text-transparent bg-clip-text bg-gradient-to-r from-secondary to-accent">está en la Biología</span>
            </h1>
            <p class="mt-4 max-w-2xl text-lg md:text-xl text-slate-600 mx-auto mb-10 leading-relaxed">
                Un framework experimental que integra los últimos avances de la neurociencia y la psicología cognitiva para superar los límites de los modelos de lenguaje (LLMs) actuales.
            </p>
            <div class="flex flex-col sm:flex-row justify-center items-center gap-4">
                <a href="#resultados" class="w-full sm:w-auto px-8 py-3.5 border border-transparent text-base font-medium rounded-xl text-white bg-primary hover:bg-blue-800 shadow-lg hover:shadow-xl transition-all transform hover:-translate-y-0.5">
                    Ver Benchmark Empírico
                </a>
                <a href="#metodologia" class="w-full sm:w-auto px-8 py-3.5 border border-slate-300 text-base font-medium rounded-xl text-slate-700 bg-white hover:bg-slate-50 shadow-sm transition-all">
                    Explorar Metodología
                </a>
            </div>
        </div>
    </section>

    <!-- 2. The Problem Section -->
    <section id="problema" class="py-20 bg-white border-y border-slate-100">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="text-center max-w-3xl mx-auto mb-16">
                <h2 class="text-3xl font-bold text-primary sm:text-4xl mb-4">Los Límites de la Fuerza Bruta</h2>
                <p class="text-lg text-slate-600 leading-relaxed">
                    La arquitectura Transformer clásica ha revolucionado el mundo tecnológico. Sin embargo, su dependencia de la atención densa (matemáticamente <strong>O(N²)</strong>) ha creado modelos con problemas estructurales insostenibles a largo plazo.
                </p>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
                <!-- Card 1 -->
                <div class="bg-slate-50 rounded-2xl p-8 border border-slate-100 shadow-sm hover:shadow-md transition-shadow group">
                    <div class="text-5xl mb-6 group-hover:scale-110 transition-transform duration-300 origin-left">⚡</div>
                    <h3 class="text-xl font-bold text-slate-800 mb-3">Ineficiencia Energética</h3>
                    <p class="text-slate-600">
                        Computación masiva y repetitiva en cada token. Un modelo tradicional gasta la misma energía procesando un artículo determinista ("el", "la") que resolviendo una ecuación matemática compleja.
                    </p>
                </div>
                <!-- Card 2 -->
                <div class="bg-slate-50 rounded-2xl p-8 border border-slate-100 shadow-sm hover:shadow-md transition-shadow group">
                    <div class="text-5xl mb-6 group-hover:scale-110 transition-transform duration-300 origin-left">🌪️</div>
                    <h3 class="text-xl font-bold text-slate-800 mb-3">Olvido Catastrófico</h3>
                    <p class="text-slate-600">
                        Incapacidad de aprender continuamente. Entrenar a un Transformer con información nueva a menudo destruye el conocimiento previo (sobreescritura de pesos), impidiendo una adaptación dinámica.
                    </p>
                </div>
                <!-- Card 3 -->
                <div class="bg-slate-50 rounded-2xl p-8 border border-slate-100 shadow-sm hover:shadow-md transition-shadow group">
                    <div class="text-5xl mb-6 group-hover:scale-110 transition-transform duration-300 origin-left">🧊</div>
                    <h3 class="text-xl font-bold text-slate-800 mb-3">Rigidez Semántica</h3>
                    <p class="text-slate-600">
                        Interferencia en el espacio vectorial. Mezclar conceptos matemáticos, tono emocional y lógica en un mismo bloque denso provoca alucinaciones cuando los contextos se cruzan de forma inesperada.
                    </p>
                </div>
            </div>
        </div>
    </section>

    <!-- 3. Architectures Section -->
    <section id="arquitecturas" class="py-20 bg-background relative">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="mb-12">
                <h2 class="text-3xl font-bold text-primary sm:text-4xl mb-4">Evolución Arquitectónica Bio-Inspirada</h2>
                <p class="text-lg text-slate-600 max-w-3xl">
                    Para solucionar las ineficiencias, nuestro framework compara empíricamente el modelo estándar frente a 7 nuevos módulos construidos a partir de descubrimientos neurocientíficos de vanguardia (2024-2026). Explora las distintas generaciones de nuestra arquitectura.
                </p>
            </div>

            <!-- Tab Navigation -->
            <div class="flex border-b border-slate-200 mb-8 overflow-x-auto no-scrollbar">
                <button onclick="switchTab('v1')" id="tab-v1" class="tab-active py-4 px-6 text-lg font-medium transition-colors whitespace-nowrap outline-none">
                    Fundamentos (v1.0)
                </button>
                <button onclick="switchTab('v2')" id="tab-v2" class="tab-inactive py-4 px-6 text-lg font-medium transition-colors whitespace-nowrap outline-none">
                    Módulos Cognitivos (v2.0)
                </button>
            </div>

            <!-- Content V1.0 -->
            <div id="content-v1" class="grid grid-cols-1 md:grid-cols-2 gap-6 transition-opacity duration-500">
                <!-- DCA -->
                <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200 cursor-pointer hover:border-secondary transition-colors" onclick="toggleDetails('dca-details')">
                    <div class="flex items-center justify-between mb-2">
                        <div class="flex items-center gap-3">
                            <span class="text-3xl">🧠</span>
                            <h3 class="text-xl font-bold text-slate-800">DCA</h3>
                        </div>
                        <span class="text-slate-400 text-sm">Ver detalles ▼</span>
                    </div>
                    <p class="text-primary font-medium text-sm mb-3">Dynamic Connectome Architecture</p>
                    <p class="text-slate-600 text-sm">Enrutamiento escaso (sparse) que imita la eficiencia de las conexiones del conectoma biológico (ej. el mapeo del cerebro de la mosca), reduciendo los FLOPs masivamente.</p>
                    
                    <div id="dca-details" class="hidden mt-4 pt-4 border-t border-slate-100 text-sm text-slate-700 bg-slate-50 p-4 rounded-lg">
                        <strong>Mecanismo Interno:</strong> Abandona la matriz densa (N x N) del Transformer. Utiliza proyecciones estructuradas dispersas. Físicamente, en la GPU, esto se traduce en operaciones `torch.sparse` o Block-Sparse, garantizando que el hardware solo calcule las "autopistas" neuronales activas.
                    </div>
                </div>

                <!-- MOPN -->
                <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200 cursor-pointer hover:border-secondary transition-colors" onclick="toggleDetails('mopn-details')">
                    <div class="flex items-center justify-between mb-2">
                        <div class="flex items-center gap-3">
                            <span class="text-3xl">📐</span>
                            <h3 class="text-xl font-bold text-slate-800">MOPN</h3>
                        </div>
                        <span class="text-slate-400 text-sm">Ver detalles ▼</span>
                    </div>
                    <p class="text-primary font-medium text-sm mb-3">Multi-dimensional Orthogonal Processing</p>
                    <p class="text-slate-600 text-sm">Proyecciones en subespacios ortogonales para compartimentar la lógica, el tono y la semántica, evitando alucinaciones por cruce de información.</p>
                    
                    <div id="mopn-details" class="hidden mt-4 pt-4 border-t border-slate-100 text-sm text-slate-700 bg-slate-50 p-4 rounded-lg">
                        <strong>Mecanismo Interno:</strong> Fuerza matemáticamente a las cabezas de atención a operar en vectores perpendiculares. El ruido semántico de un concepto no afecta al cálculo lógico de otro, permitiendo un "control de mandos" preciso sobre la salida.
                    </div>
                </div>

                <!-- SCT -->
                <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200 cursor-pointer hover:border-secondary transition-colors" onclick="toggleDetails('sct-details')">
                    <div class="flex items-center justify-between mb-2">
                        <div class="flex items-center gap-3">
                            <span class="text-3xl">💤</span>
                            <h3 class="text-xl font-bold text-slate-800">SCT</h3>
                        </div>
                        <span class="text-slate-400 text-sm">Ver detalles ▼</span>
                    </div>
                    <p class="text-primary font-medium text-sm mb-3">Sleep-Cycled Transformers</p>
                    <p class="text-slate-600 text-sm">Algoritmo de consolidación metabólica y poda de pesos <i>offline</i> que imita el sueño humano para permitir un aprendizaje continuo.</p>
                    
                    <div id="sct-details" class="hidden mt-4 pt-4 border-t border-slate-100 text-sm text-slate-700 bg-slate-50 p-4 rounded-lg">
                        <strong>Mecanismo Interno:</strong> Posee una memoria a corto plazo activa durante la inferencia. En ciclos inactivos ("sueño"), ejecuta una limpieza de gradientes y poda conexiones débiles, fusionando el conocimiento nuevo con el antiguo sin sufrir olvido catastrófico.
                    </div>
                </div>

                <!-- GMA-MoE -->
                <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200 cursor-pointer hover:border-secondary transition-colors" onclick="toggleDetails('gma-details')">
                    <div class="flex items-center justify-between mb-2">
                        <div class="flex items-center gap-3">
                            <span class="text-3xl">🛡️</span>
                            <h3 class="text-xl font-bold text-slate-800">GMA-MoE</h3>
                        </div>
                        <span class="text-slate-400 text-sm">Ver detalles ▼</span>
                    </div>
                    <p class="text-primary font-medium text-sm mb-3">Glial Modulation & Mixture of Experts</p>
                    <p class="text-slate-600 text-sm">Red glial paralela que actúa como gestor energético, asignando parámetros dinámicamente según la complejidad (el estrés) de la instrucción.</p>
                    
                    <div id="gma-details" class="hidden mt-4 pt-4 border-t border-slate-100 text-sm text-slate-700 bg-slate-50 p-4 rounded-lg">
                        <strong>Mecanismo Interno:</strong> Una red secundaria diminuta monitorea el "estrés" de pérdida (entropía). Si la tarea es simple, activa un solo experto (ahorro masivo). Si es compleja, despliega toda la potencia del enjambre de expertos.
                    </div>
                </div>
            </div>

            <!-- Content V2.0 (Hidden by default) -->
            <div id="content-v2" class="hidden grid-cols-1 lg:grid-cols-3 gap-6 transition-opacity duration-500">
                <!-- PMT -->
                <div class="bg-white rounded-xl p-6 shadow-md border-t-4 border-accent cursor-pointer hover:-translate-y-1 transition-transform" onclick="toggleDetails('pmt-details')">
                    <div class="text-4xl mb-4">⚡</div>
                    <h3 class="text-xl font-bold text-slate-800 mb-1">PMT</h3>
                    <p class="text-emerald-600 font-semibold text-sm mb-3">Predictive Minimalist Trace</p>
                    <p class="text-slate-600 text-sm">Sistema de salida temprana (Early Exit) a nivel de token. La red solo consume energía profunda en palabras que generan "sorpresa".</p>
                    <div id="pmt-details" class="hidden mt-4 pt-4 border-t border-slate-100 text-sm text-slate-700 bg-slate-50 p-3 rounded">
                        Mediante el enmascaramiento de tokens, congela representaciones predecibles en las primeras capas, evitando procesamientos inútiles en las capas superiores.
                    </div>
                </div>

                <!-- CEN -->
                <div class="bg-white rounded-xl p-6 shadow-md border-t-4 border-accent cursor-pointer hover:-translate-y-1 transition-transform" onclick="toggleDetails('cen-details')">
                    <div class="text-4xl mb-4">🔮</div>
                    <h3 class="text-xl font-bold text-slate-800 mb-1">CEN</h3>
                    <p class="text-emerald-600 font-semibold text-sm mb-3">Counterfactual Episodic Network</p>
                    <p class="text-slate-600 text-sm">Simulación latente de escenarios múltiples. La IA piensa "¿Y si...?" e imagina futuros posibles antes de generar un output crítico.</p>
                    <div id="cen-details" class="hidden mt-4 pt-4 border-t border-slate-100 text-sm text-slate-700 bg-slate-50 p-3 rounded">
                        Integra un árbol de búsqueda latente directamente en la capa de atención, evaluando la coherencia semántica de múltiples respuestas antes de comprometerse con un token.
                    </div>
                </div>

                <!-- VLM -->
                <div class="bg-white rounded-xl p-6 shadow-md border-t-4 border-accent cursor-pointer hover:-translate-y-1 transition-transform" onclick="toggleDetails('vlm-details')">
                    <div class="text-4xl mb-4">👥</div>
                    <h3 class="text-xl font-bold text-slate-800 mb-1">VLM</h3>
                    <p class="text-emerald-600 font-semibold text-sm mb-3">Vicarious Learning Module</p>
                    <p class="text-slate-600 text-sm">Una red estudiante independiente que aprende "en la sombra", imitando la lógica del maestro sin perturbar sus gradientes.</p>
                    <div id="vlm-details" class="hidden mt-4 pt-4 border-t border-slate-100 text-sm text-slate-700 bg-slate-50 p-3 rounded">
                        Usa la función <code>.detach()</code> aislando matemáticamente al estudiante. Permite la destilación del conocimiento en tiempo real sin el costoso proceso de Fine-Tuning offline.
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- 4. Methodology Section -->
    <section id="metodologia" class="py-20 bg-white border-y border-slate-100">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="flex flex-col lg:flex-row gap-12 items-center">
                <div class="w-full lg:w-1/2">
                    <h2 class="text-3xl font-bold text-primary sm:text-4xl mb-6">Diseñado para la Validación Empírica</h2>
                    <p class="text-lg text-slate-600 mb-8 leading-relaxed">
                        Para asegurar la máxima integridad científica y evitar la pseudociencia matemática de modelos simulados, el framework <code class="bg-slate-100 px-2 py-1 rounded text-sm text-secondary">jacoboariza/neuro_transformer</code> exige el cumplimiento de 5 requisitos técnicos estrictos.
                    </p>
                    
                    <div class="bg-slate-50 p-6 rounded-2xl border border-slate-200">
                        <div class="flex justify-between items-center mb-4">
                            <span class="font-bold text-slate-800">Progreso de Auditoría</span>
                            <span id="audit-progress-text" class="text-secondary font-bold">0%</span>
                        </div>
                        <div class="w-full bg-slate-200 rounded-full h-2.5 mb-6">
                            <div id="audit-progress-bar" class="bg-secondary h-2.5 rounded-full transition-all duration-500" style="width: 0%"></div>
                        </div>
                        <p class="text-sm text-slate-500 italic">Haz clic en los requisitos de la derecha para verificar su implementación en la base de código.</p>
                    </div>
                </div>

                <div class="w-full lg:w-1/2">
                    <div class="space-y-4">
                        <!-- Checklist Items -->
                        <div class="checklist-item flex items-start gap-4 p-4 border border-slate-200 rounded-xl cursor-pointer hover:bg-slate-50 transition-colors" onclick="toggleCheck(this, 0)">
                            <div class="check-circle w-6 h-6 rounded-full border-2 border-slate-300 flex items-center justify-center flex-shrink-0 mt-0.5 transition-colors"></div>
                            <div>
                                <h4 class="font-bold text-slate-800">Entrenamiento con Datos Reales</h4>
                                <p class="text-sm text-slate-600 mt-1">Uso de Datasets reales (<code class="text-xs text-blue-600">HuggingFaceFW/fineweb-edu</code>) y tokenización real (SmolLM). Prohibido el uso de ruido aleatorio.</p>
                            </div>
                        </div>

                        <div class="checklist-item flex items-start gap-4 p-4 border border-slate-200 rounded-xl cursor-pointer hover:bg-slate-50 transition-colors" onclick="toggleCheck(this, 1)">
                            <div class="check-circle w-6 h-6 rounded-full border-2 border-slate-300 flex items-center justify-center flex-shrink-0 mt-0.5 transition-colors"></div>
                            <div>
                                <h4 class="font-bold text-slate-800">Escasez de Hardware Real</h4>
                                <p class="text-sm text-slate-600 mt-1">Implementación nativa con <code class="text-xs text-blue-600">torch.sparse</code>, garantizando un ahorro auténtico en VRAM, no solo una máscara matemática multiplicativa.</p>
                            </div>
                        </div>

                        <div class="checklist-item flex items-start gap-4 p-4 border border-slate-200 rounded-xl cursor-pointer hover:bg-slate-50 transition-colors" onclick="toggleCheck(this, 2)">
                            <div class="check-circle w-6 h-6 rounded-full border-2 border-slate-300 flex items-center justify-center flex-shrink-0 mt-0.5 transition-colors"></div>
                            <div>
                                <h4 class="font-bold text-slate-800">Aislamiento de Gradientes</h4>
                                <p class="text-sm text-slate-600 mt-1">Uso estricto de <code class="text-xs text-blue-600">.detach()</code> para evitar la contaminación cruzada en el backpropagation del aprendizaje vicario.</p>
                            </div>
                        </div>

                        <div class="checklist-item flex items-start gap-4 p-4 border border-slate-200 rounded-xl cursor-pointer hover:bg-slate-50 transition-colors" onclick="toggleCheck(this, 3)">
                            <div class="check-circle w-6 h-6 rounded-full border-2 border-slate-300 flex items-center justify-center flex-shrink-0 mt-0.5 transition-colors"></div>
                            <div>
                                <h4 class="font-bold text-slate-800">Enmascaramiento de Tokens</h4>
                                <p class="text-sm text-slate-600 mt-1">El mecanismo PMT funciona aislando tokens específicos, manteniendo la coherencia de la secuencia en procesamientos en lote (Batch).</p>
                            </div>
                        </div>

                        <div class="checklist-item flex items-start gap-4 p-4 border border-slate-200 rounded-xl cursor-pointer hover:bg-slate-50 transition-colors" onclick="toggleCheck(this, 4)">
                            <div class="check-circle w-6 h-6 rounded-full border-2 border-slate-300 flex items-center justify-center flex-shrink-0 mt-0.5 transition-colors"></div>
                            <div>
                                <h4 class="font-bold text-slate-800">Profiling Avanzado</h4>
                                <p class="text-sm text-slate-600 mt-1">Tiempos cronometrados con precisión mediante eventos nativos de hardware en GPU (<code class="text-xs text-blue-600">torch.cuda.Event</code>).</p>
                            </div>
                        </div>

                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- 5. Benchmark Results Section -->
    <section id="resultados" class="py-20 bg-background">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="text-center max-w-3xl mx-auto mb-16">
                <span class="inline-block py-1 px-3 rounded-full bg-emerald-100 text-emerald-800 text-sm font-semibold tracking-wider mb-4 border border-emerald-200">
                    DATOS EMPÍRICOS
                </span>
                <h2 class="text-3xl font-bold text-primary sm:text-4xl mb-4">Eficiencia vs Precisión: Los Resultados</h2>
                <p class="text-lg text-slate-600 leading-relaxed">
                    Evaluación comparativa del Transformer Baseline contra nuestras arquitecturas. Descubre qué combinaciones ofrecen la mejor relación precisión-eficiencia (Trade-off).
                </p>
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-2 gap-10">
                <!-- Chart 1: Inference Time -->
                <div class="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                    <h3 class="text-xl font-bold text-slate-800 mb-2">Velocidad de Inferencia</h3>
                    <p class="text-sm text-slate-500 mb-6">Tiempo de procesamiento (en segundos) para la misma carga de trabajo. Menor es mejor.</p>
                    <div class="chart-container">
                        <canvas id="timeChart"></canvas>
                    </div>
                    <div class="mt-4 text-sm text-slate-600 bg-blue-50 p-3 rounded-lg border border-blue-100">
                        <strong>Insight:</strong> El modelo DCA (Conectoma) logra una reducción del tiempo a la mitad (~1.21s) en comparación con la atención densa tradicional, logrando casi medio millón de tokens por segundo.
                    </div>
                </div>

                <!-- Chart 2: Scatter Plot Loss vs Params -->
                <div class="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                    <h3 class="text-xl font-bold text-slate-800 mb-2">Compromiso: Parámetros vs Loss</h3>
                    <p class="text-sm text-slate-500 mb-6">Pérdida cruzada (Eje Y, menor es mejor) en función del número de parámetros. Eje X en miles.</p>
                    <div class="chart-container">
                        <canvas id="scatterChart"></canvas>
                    </div>
                    <div class="mt-4 text-sm text-slate-600 bg-emerald-50 p-3 rounded-lg border border-emerald-100">
                        <strong>Insight:</strong> MOPN y DCA reducen significativamente el peso computacional (~25% menos parámetros) manteniendo una pérdida estructural casi idéntica a la del Baseline rígido.
                    </div>
                </div>
            </div>
            
            <div class="mt-12 text-center">
                <p class="text-slate-600 mb-6 max-w-2xl mx-auto">
                    Los resultados confirman que la especialización estructural (inspirada en la escasez y ortogonalidad cortical) puede superar el enfoque contemporáneo de escalado por fuerza bruta.
                </p>
            </div>
        </div>
    </section>

    <!-- 6. Footer -->
    <footer class="bg-slate-900 text-white py-12 border-t border-slate-800">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="grid grid-cols-1 md:grid-cols-3 gap-8 items-center text-center md:text-left">
                
                <div class="col-span-1 md:col-span-2">
                    <h3 class="text-2xl font-bold text-white mb-2 flex items-center justify-center md:justify-start">
                        <span class="mr-2">🧠</span> Neuro Transformer
                    </h3>
                    <p class="text-slate-400 max-w-md mx-auto md:mx-0">
                        ¿Listo para construir la próxima generación de modelos de lenguaje eficientes y biológicamente plausibles?
                    </p>
                </div>
                
                <div class="col-span-1 flex justify-center md:justify-end">
                    <a href="https://github.com/jacoboariza/neuro_transformer" target="_blank" rel="noopener noreferrer" class="bg-secondary hover:bg-blue-600 text-white px-6 py-3 rounded-xl font-medium transition-colors shadow-lg flex items-center gap-2">
                        <span>Clonar el Repositorio</span>
                        <span class="text-xl">↗</span>
                    </a>
                </div>
            </div>
            
            <div class="mt-12 pt-8 border-t border-slate-800 flex flex-col md:flex-row justify-between items-center text-sm text-slate-500">
                <p>© 2026 Proyecto Neuro Transformer. Licencia MIT.</p>
                <div class="flex space-x-6 mt-4 md:mt-0">
                    <a href="#" class="hover:text-white transition-colors">Documentación</a>
                    <a href="#" class="hover:text-white transition-colors">Autor (jacoboariza)</a>
                    <a href="#" class="hover:text-white transition-colors">Contacto</a>
                </div>
            </div>
        </div>
    </footer>

    <!-- JS Logic -->
    <script>
        // --- Navigation Logic ---
        const mobileMenuBtn = document.getElementById('mobile-menu-btn');
        const mobileMenu = document.getElementById('mobile-menu');

        mobileMenuBtn.addEventListener('click', () => {
            mobileMenu.classList.toggle('hidden');
        });

        // Close mobile menu on click
        document.querySelectorAll('#mobile-menu a').forEach(link => {
            link.addEventListener('click', () => {
                mobileMenu.classList.add('hidden');
            });
        });

        // --- Tabs Logic (v1 vs v2) ---
        function switchTab(tabId) {
            const v1Tab = document.getElementById('tab-v1');
            const v2Tab = document.getElementById('tab-v2');
            const v1Content = document.getElementById('content-v1');
            const v2Content = document.getElementById('content-v2');

            if (tabId === 'v1') {
                v1Tab.className = 'tab-active py-4 px-6 text-lg font-medium transition-colors whitespace-nowrap outline-none';
                v2Tab.className = 'tab-inactive py-4 px-6 text-lg font-medium transition-colors whitespace-nowrap outline-none';
                v1Content.classList.remove('hidden');
                v1Content.classList.add('grid');
                v2Content.classList.add('hidden');
                v2Content.classList.remove('grid');
            } else {
                v2Tab.className = 'tab-active py-4 px-6 text-lg font-medium transition-colors whitespace-nowrap outline-none';
                v1Tab.className = 'tab-inactive py-4 px-6 text-lg font-medium transition-colors whitespace-nowrap outline-none';
                v2Content.classList.remove('hidden');
                v2Content.classList.add('grid');
                v1Content.classList.add('hidden');
                v1Content.classList.remove('grid');
            }
        }

        // --- Details Toggle (Cards) ---
        function toggleDetails(id) {
            const el = document.getElementById(id);
            if (el.classList.contains('hidden')) {
                el.classList.remove('hidden');
                el.classList.add('block', 'animate-fade-in');
            } else {
                el.classList.add('hidden');
                el.classList.remove('block', 'animate-fade-in');
            }
        }

        // --- Interactive Methodology Checklist ---
        let checks = [false, false, false, false, false];
        
        function toggleCheck(element, index) {
            checks[index] = !checks[index];
            const circle = element.querySelector('.check-circle');
            
            if (checks[index]) {
                circle.classList.remove('border-slate-300');
                circle.classList.add('bg-secondary', 'border-secondary');
                // Using unicode checkmark as SVG is forbidden
                circle.innerHTML = '<span class="text-white font-bold text-sm">✓</span>';
                element.classList.add('bg-blue-50', 'border-blue-200');
                element.classList.remove('border-slate-200');
            } else {
                circle.classList.add('border-slate-300');
                circle.classList.remove('bg-secondary', 'border-secondary');
                circle.innerHTML = '';
                element.classList.remove('bg-blue-50', 'border-blue-200');
                element.classList.add('border-slate-200');
            }
            updateProgress();
        }

        function updateProgress() {
            const completed = checks.filter(c => c).length;
            const percentage = (completed / checks.length) * 100;
            
            document.getElementById('audit-progress-bar').style.width = percentage + '%';
            document.getElementById('audit-progress-text').innerText = Math.round(percentage) + '%';
        }

        // --- Chart.js Implementations ---
        document.addEventListener('DOMContentLoaded', function() {
            
            // Common Chart Options for Responsiveness
            const commonOptions = {
                responsive: true,
                maintainAspectRatio: false, // CRITICAL: allows the chart to fill the container defined by CSS
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { font: { family: "'Inter', sans-serif" } }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(15, 23, 42, 0.9)',
                        titleFont: { family: "'Inter', sans-serif", size: 14 },
                        bodyFont: { family: "'Inter', sans-serif", size: 13 },
                        padding: 12,
                        cornerRadius: 8,
                        callbacks: {
                            label: function(context) {
                                // Wrapping logic or specific formatting
                                let label = context.dataset.label || '';
                                if (label) { label += ': '; }
                                if (context.parsed.y !== null) { label += context.parsed.y; }
                                return label;
                            }
                        }
                    }
                }
            };

            // 1. Bar Chart: Inference Time
            const ctxTime = document.getElementById('timeChart').getContext('2d');
            new Chart(ctxTime, {
                type: 'bar',
                data: {
                    labels: ['Transformer Base', 'DCA (Sparse)', 'MOPN (Ortho)'],
                    datasets: [{
                        label: 'Segundos de Entrenamiento (Menor es mejor)',
                        data: [2.77, 1.21, 1.33],
                        backgroundColor: [
                            'rgba(148, 163, 184, 0.8)', // Slate (Baseline)
                            'rgba(59, 130, 246, 0.8)',  // Blue (DCA)
                            'rgba(16, 185, 129, 0.8)'   // Emerald (MOPN)
                        ],
                        borderColor: [
                            'rgb(100, 116, 139)',
                            'rgb(37, 99, 235)',
                            'rgb(5, 150, 105)'
                        ],
                        borderWidth: 1,
                        borderRadius: 6
                    }]
                },
                options: {
                    ...commonOptions,
                    scales: {
                        y: { 
                            beginAtZero: true,
                            grid: { color: '#f1f5f9' },
                            title: { display: true, text: 'Segundos (s)' }
                        },
                        x: { grid: { display: false } }
                    }
                }
            });

            // 2. Scatter Plot: Trade-off (Loss vs Parameters)
            const ctxScatter = document.getElementById('scatterChart').getContext('2d');
            new Chart(ctxScatter, {
                type: 'scatter',
                data: {
                    datasets: [
                        {
                            label: 'Transformer Base',
                            data: [{x: 178.9, y: 6.55}],
                            backgroundColor: 'rgb(100, 116, 139)',
                            pointRadius: 8,
                            pointHoverRadius: 10
                        },
                        {
                            label: 'DCA',
                            data: [{x: 137.3, y: 6.66}],
                            backgroundColor: 'rgb(37, 99, 235)',
                            pointRadius: 8,
                            pointHoverRadius: 10
                        },
                        {
                            label: 'MOPN',
                            data: [{x: 134.2, y: 6.65}],
                            backgroundColor: 'rgb(5, 150, 105)',
                            pointRadius: 8,
                            pointHoverRadius: 10
                        }
                    ]
                },
                options: {
                    ...commonOptions,
                    plugins: {
                        ...commonOptions.plugins,
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return `${context.dataset.label}: ${context.parsed.x}k params, Loss: ${context.parsed.y}`;
                                }
                            }
                        }
                    },
                    scales: {
                        y: {
                            title: { display: true, text: 'Loss (Error Final)' },
                            min: 6.5,
                            max: 6.7,
                            grid: { color: '#f1f5f9' }
                        },
                        x: {
                            title: { display: true, text: 'Parámetros (Miles)' },
                            min: 130,
                            max: 185,
                            grid: { color: '#f1f5f9' }
                        }
                    }
                }
            });
        });
    </script>
</body>
</html>