import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from mga_algorithm import OptimizedMGA


DATASET_NAME = 'SRBCT' #aqui muda conforme o dataset que tu quer fazer o grafico
DATA_PATH = f"data/biomedical/{DATASET_NAME}.mat"

POPULATION_SIZE = 50
MAX_ITERATIONS = 100  
K_NEIGHBORS = 5       

def load_data(path):
    print(f"\n--- Analisando arquivo: {path} ---")
    if not os.path.exists(path):
        raise FileNotFoundError(f"ERRO: Arquivo não encontrado em {path}")
        
    data = scipy.io.loadmat(path)
    keys = [k for k in data.keys() if not k.startswith('__')]
    
    X = None
    y = None
    
    
    arrays = []
    for k in keys:
        val = data[k]
        if isinstance(val, np.ndarray):
            arrays.append((k, val))
    arrays.sort(key=lambda x: x[1].size, reverse=True)
    
    if len(arrays) > 0:
        X_name, X = arrays[0]
        print(f" -> Matriz principal '{X_name}' detectada com shape {X.shape}")
    else:
        raise ValueError("Arquivo vazio.")

    
    possible_y_names = ['Y', 'y', 'labels', 'class', 'L']
    for name in possible_y_names:
        for k in keys:
            if k.lower() == name.lower() and k != X_name:
                y = data[k]
                break
        if y is not None: break

    
    if y is None:
        print(" -> y não encontrado separadamente. Procurando dentro de X...")
        
        
        last_col = X[:, -1]
        unique_last = np.unique(last_col)
        
        
        first_col = X[:, 0]
        unique_first = np.unique(first_col)
        
        print(f"   (Debug) Última coluna tem {len(unique_last)} valores únicos.")
        print(f"   (Debug) Primeira coluna tem {len(unique_first)} valores únicos.")

        
        if len(unique_last) < 20:
            print(f" -> SUCESSO! Labels encontrados na ÚLTIMA coluna.")
            y = last_col
            X = X[:, :-1] 
        elif len(unique_first) < 20:
            print(f" -> SUCESSO! Labels encontrados na PRIMEIRA coluna.")
            y = first_col
            X = X[:, 1:] 
        else:
            
            print(f"   (Erro) Amostra da última coluna: {last_col[:5]}")
            raise ValueError("Não foi possível distinguir dados de labels automaticamente.")

    y = np.array(y).flatten()
    return X, y

def main():
    print(f"--- Iniciando {DATASET_NAME} ---")
    
    try:
        X, y = load_data(DATA_PATH)
        print(f"Dados Carregados: {X.shape[0]} amostras, {X.shape[1]} features.")
    except Exception as e:
        print(str(e))
        return

    print("\nInicializando MGA")
    mga = OptimizedMGA(
        X=X, 
        y=y, 
        population_size=POPULATION_SIZE, 
        max_iter=MAX_ITERATIONS, 
        k_neighbors=K_NEIGHBORS
    )

    print("Rodando Evolução")
    final_population, history = mga.run()
    
    best_accuracy = max(history)
    print(f"\n--- FIM DA EXECUÇÃO ---")
    print(f"Melhor Acurácia Atingida: {best_accuracy*100:.2f}%")

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(history)), history, color='red', linewidth=2, label='MGA')
    
    plt.title(f'Convergência MGA - {DATASET_NAME.upper()}', fontsize=14)
    plt.xlabel('Iterações (Gerações)', fontsize=12)
    plt.ylabel('Acurácia (Fitness)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    

    plt.savefig(f"resultado_replicacao_{DATASET_NAME}.png")
    print(f"Gráfico salvo como 'resultado_replicacao_{DATASET_NAME}.png'")
    plt.show()

if __name__ == "__main__":
    main()