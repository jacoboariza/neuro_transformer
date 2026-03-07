from caveclient import CAVEclient
import networkx as nx
import numpy as np

# 1. Inicializar el cliente para el dataset oficial de FlyWire
client = CAVEclient('flywire_fafb_production')

# Nota: La primera vez que ejecutes esto, CAVEclient te pedirá que introduzcas 
# tu token de autenticación. Sigue las instrucciones de la consola.

# 2. Definir las neuronas objetivo
# En FlyWire, cada neurona tiene un identificador único llamado "Root ID" (un número muy largo).
# En un proyecto real, consultarías la tabla de tipos celulares para obtener todos los IDs 
# del "Mushroom Body". Aquí usaremos unos IDs de ejemplo para ilustrar:
mis_neuronas = [720575940628855011, 720575940612345678, 720575940698765432] 

print("Descargando sinapsis desde FlyWire...")

# 3. Consultar las conexiones físicas (sinapsis)
# Pedimos a la base de datos todas las conexiones donde nuestras neuronas 
# actúan tanto de emisoras (pre) como de receptoras (post).
sinapsis_df = client.materialize.query_table(
    'synapses_nt_v120', # Tabla oficial de sinapsis
    filter_in_dict={
        'pre_pt_root_id': mis_neuronas,
        'post_pt_root_id': mis_neuronas
    }
)

# 4. Construir el grafo neuronal
G = nx.DiGraph() # Grafo Dirigido (la información fluye en una dirección)

# Iterar sobre las sinapsis y construir las conexiones
for index, row in sinapsis_df.iterrows():
    pre_id = row['pre_pt_root_id']
    post_id = row['post_pt_root_id']
    
    # Si la conexión ya existe, sumamos "fuerza" (peso) a la sinapsis
    if G.has_edge(pre_id, post_id):
        G[pre_id][post_id]['weight'] += 1
    else:
        G.add_edge(pre_id, post_id, weight=1)

# 5. Convertir a matriz matemática para IA
# Esto genera una matriz NumPy. Si hay conexión, el valor > 0. Si no, es 0.
matriz_adyacencia = nx.to_numpy_array(G, nodelist=mis_neuronas)

print(f"¡Éxito! Forma de la matriz: {matriz_adyacencia.shape}")
print("Muestra de la matriz (Pesos sinápticos):")
print(matriz_adyacencia)