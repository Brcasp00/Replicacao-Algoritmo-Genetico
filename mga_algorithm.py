import numpy as np
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

class OptimizedMGA:
    def __init__(self, X, y, population_size=50, max_iter=100, k_neighbors=5):
        self.X = X
        self.y = y
        self.pop_size = population_size
        self.n_features = X.shape[1]
        self.max_iter = max_iter
        self.knn = KNeighborsClassifier(n_neighbors=k_neighbors)
        
        
        self.history_best_fitness = [] 

    def initialize_population(self):
        
        population = np.zeros((self.pop_size, self.n_features), dtype=int)
        
        for i in range(self.pop_size):
            
            n_active = random.randint(5, int(self.n_features * 0.05)) 
            if n_active < 1: n_active = 1
            
            indices = random.sample(range(self.n_features), n_active)
            population[i, indices] = 1
            
        return population

    def calculate_fitness(self, population):
        
        fitness_scores = []
        
        _, counts = np.unique(self.y, return_counts=True)
        min_samples = np.min(counts)
        
        n_splits = max(2, min(5, min_samples))
        
        for i in range(self.pop_size):
            individual = population[i]
            
            if np.sum(individual) == 0:
                fitness_scores.append(0)
                continue
            
            cols = [idx for idx, val in enumerate(individual) if val == 1]
            X_subset = self.X[:, cols]
            
            try:
                scores = cross_val_score(self.knn, X_subset, self.y, cv=n_splits)
                fitness_scores.append(scores.mean())
            except ValueError:
                fitness_scores.append(0)
            
        return np.array(fitness_scores)

    def evolve(self, population, fitness):
        next_population = []
        
        best_idx = np.argmax(fitness)
        next_population.append(population[best_idx].copy())
        
        while len(next_population) < self.pop_size:
            parent1 = self._tournament_selection(population, fitness)
            parent2 = self._tournament_selection(population, fitness)
            
            
            child1, child2 = self._uniform_crossover(parent1, parent2)
            
            
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            
            next_population.append(child1)
            if len(next_population) < self.pop_size:
                next_population.append(child2)
                
        return np.array(next_population)

    def _tournament_selection(self, population, fitness, k=3):
        candidates_idx = np.random.randint(0, self.pop_size, k)
        best_idx = candidates_idx[np.argmax(fitness[candidates_idx])]
        return population[best_idx]

    def _uniform_crossover(self, p1, p2, rate=0.8):
        if np.random.rand() > rate:
            return p1.copy(), p2.copy()
        
        mask = np.random.randint(0, 2, size=self.n_features).astype(bool)
        c1, c2 = p1.copy(), p2.copy()
        c1[mask] = p2[mask]
        c2[mask] = p1[mask]
        return c1, c2

    def _mutate(self, individual, rate=0.01):
        
        mutation_mask = np.random.rand(self.n_features) < rate
        individual[mutation_mask] = 1 - individual[mutation_mask]
        return individual

    def run(self):
        population = self.initialize_population()
        
        for iteration in range(self.max_iter):
            fitness = self.calculate_fitness(population)
            
            best_val = np.max(fitness)
            self.history_best_fitness.append(best_val)
            
            
            print(f"Iteração {iteration+1}/{self.max_iter} | Melhor Acurácia: {best_val:.4f} | Features: {np.sum(population[np.argmax(fitness)])}")
            
            population = self.evolve(population, fitness)
            
        return population, self.history_best_fitness