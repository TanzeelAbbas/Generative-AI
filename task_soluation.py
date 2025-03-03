import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
import warnings

class Word2Vec:
    def __init__(self, sentences, vector_size=100, window=5, learning_rate=0.05, epochs=10):
        self.vector_size = vector_size
        self.window = window
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.vocab = set()
        self.word2id = {}
        self.id2word = {}
        self.W = None

        self.build_vocab(sentences)
        self.train(sentences)

    def build_vocab(self, sentences):
        word_counts = {}
        for sentence in sentences:
            for word in sentence:
                if word not in word_counts:
                    word_counts[word] = 0
                word_counts[word] += 1
                self.vocab.add(word)
        self.vocab = list(self.vocab)
        for i, word in enumerate(self.vocab):
            self.word2id[word] = i
            self.id2word[i] = word
        self.W = np.random.uniform(-0.8, 0.8, (len(self.vocab), self.vector_size))

    def train(self, sentences):
        for _ in range(self.epochs):
            for sentence in sentences:
                for i, target_word in enumerate(sentence):
                    target_index = self.word2id[target_word]
                    for j in range(max(0, i - self.window), min(len(sentence), i + self.window + 1)):
                        if i != j:
                            context_word = sentence[j]
                            context_index = self.word2id[context_word]
                            score = self.W[target_index].dot(self.W[context_index])
                            pred = self.sigmoid(score)
                            error = pred - 1  # For positive samples
                            self.W[target_index] -= self.learning_rate * error * self.W[context_index]
                            self.W[context_index] -= self.learning_rate * error * self.W[target_index]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def get_word_vector(self, word):
        if word in self.word2id:
            return self.W[self.word2id[word]]
        else:
            raise KeyError("Word not in vocabulary.")

class GloVe:
    def __init__(self, sentences, vector_size=100, window=5, learning_rate=0.05, epochs=10):
        self.vector_size = vector_size
        self.window = window
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.vocab = set()
        self.word2id = {}
        self.id2word = {}
        self.W = None

        self.build_vocab(sentences)
        self.train(sentences)

    def build_vocab(self, sentences):
        word_counts = {}
        for sentence in sentences:
            for word in sentence:
                if word not in word_counts:
                    word_counts[word] = 0
                word_counts[word] += 1
                self.vocab.add(word)
        self.vocab = list(self.vocab)
        for i, word in enumerate(self.vocab):
            self.word2id[word] = i
            self.id2word[i] = word
        self.W = np.random.uniform(-0.8, 0.8, (len(self.vocab), self.vector_size))

    def train(self, sentences):
        co_occurrence_matrix = np.zeros((len(self.vocab), len(self.vocab)))
        for sentence in sentences:
            for i, target_word in enumerate(sentence):
                target_index = self.word2id[target_word]
                for j in range(max(0, i - self.window), min(len(sentence), i + self.window + 1)):
                    if i != j:
                        context_word = sentence[j]
                        context_index = self.word2id[context_word]
                        co_occurrence_matrix[target_index][context_index] += 1

        for _ in range(self.epochs):
            for i in range(len(self.vocab)):
                for j in range(len(self.vocab)):
                    if co_occurrence_matrix[i][j] != 0:
                        x_ij = co_occurrence_matrix[i][j]
                        score = np.dot(self.W[i], self.W[j])
                        error = score - np.log(x_ij)
                        self.W[i] -= self.learning_rate * error * self.W[j]
                        self.W[j] -= self.learning_rate * error * self.W[i]

    def get_word_vector(self, word):
        if word in self.word2id:
            return self.W[self.word2id[word]]
        else:
            raise KeyError("Word not in vocabulary.")

class MulticlassDebiasing:
    def __init__(self, word_embeddings):
        self.word_embeddings = word_embeddings
        self.bias_subspace = None

    def identify_bias_subspace(self, defining_sets, num_components):
        mean_embeddings = []
        for defining_set in defining_sets:
            mean_embedding = np.mean([self.word_embeddings[word] for word in defining_set if word in self.word_embeddings], axis=0)
            if not np.isnan(mean_embedding).any():
                mean_embeddings.append(mean_embedding)
        mean_embeddings = np.array(mean_embeddings)

        if len(mean_embeddings) < num_components:
            num_components = len(mean_embeddings)
            print("Reducing number of components to {} due to insufficient samples.".format(num_components))

        if num_components > 0:
            pca = PCA(n_components=num_components)
            self.bias_subspace = pca.fit(mean_embeddings).components_
        else:
            print("No valid samples available to perform PCA.")

    def remove_bias_components(self, equality_sets, method='hard'):
        if self.bias_subspace is None:
            print("Bias subspace not identified. Please run identify_bias_subspace first.")
            return
        
        if method == 'hard':
            self.hard_debias(equality_sets)
        elif method == 'soft':
            self.soft_debias(equality_sets)
        else:
            raise ValueError("Unknown debiasing method. Choose 'hard' or 'soft'.")

    def hard_debias(self, equality_sets):
        if self.bias_subspace is None:
            print("Bias subspace not identified. Please run identify_bias_subspace first.")
            return
        
        for equality_set in equality_sets:
            mean_embedding = np.mean([self.word_embeddings[word] for word in equality_set if word in self.word_embeddings], axis=0)
            if mean_embedding.size == 0:
                continue
            mean_bias_component = np.dot(mean_embedding, self.bias_subspace.T) @ self.bias_subspace
            for word in equality_set:
                if word in self.word_embeddings:
                    self.word_embeddings[word] -= mean_bias_component

    def soft_debias(self, equality_sets, lambd=0.2):
        if self.bias_subspace is None:
            print("Bias subspace not identified. Please run identify_bias_subspace first.")
            return
        
        W = np.array([self.word_embeddings[word] for word in self.word_embeddings.keys() if word in self.word_embeddings])
        N = np.array([self.word_embeddings[word] for equality_set in equality_sets for word in equality_set if word in self.word_embeddings])

        A = np.linalg.lstsq(N.T @ N + lambd * self.bias_subspace.T @ self.bias_subspace, N.T @ W, rcond=None)[0]
        self.word_embeddings = {word: A.T @ self.word_embeddings[word] for word in self.word_embeddings.keys() if word in self.word_embeddings}

    def compute_MAC(self, target_set, attribute_sets):
        similarities = []
        for target_word in target_set:
            if target_word not in self.word_embeddings:
                continue
            for attribute_set in attribute_sets:
                similarity = np.mean([1 - cosine(self.word_embeddings[target_word], self.word_embeddings[attr]) for attr in attribute_set if attr in self.word_embeddings])
                similarities.append(similarity)
        if len(similarities) == 0:
            return 0
        return np.mean(similarities)

    def evaluate_debiasing_effect(self, target_set, attribute_sets, num_permutations=1000):
        biased_MAC = self.compute_MAC(target_set, attribute_sets)

        debiased_embeddings = self.word_embeddings
        debiased_MAC = self.compute_MAC(target_set, attribute_sets)

        differences = []
        for _ in range(num_permutations):
            permuted_target_set = np.random.permutation(target_set)
            permuted_MAC = self.compute_MAC(permuted_target_set, attribute_sets)
            differences.append(permuted_MAC - debiased_MAC)

        p_value = (np.sum(np.abs(differences) >= np.abs(biased_MAC - debiased_MAC)) + 1) / (num_permutations + 1)

        return biased_MAC, debiased_MAC, p_value

# Example usage:
sentences = [['king', 'queen', 'royal'], ['man', 'woman', 'human'], ['apple', 'orange', 'fruit']]

# Initialize and train Word2Vec model
word2vec_model = Word2Vec(sentences, learning_rate=0.01)
word2vec_embeddings = {word: word2vec_model.get_word_vector(word) for word in word2vec_model.vocab}

defining_sets = [['king', 'queen'], ['man', 'woman']]  # Adjust defining sets based on available vocabulary
equality_sets = [['king'], ['queen']]
target_set = ['queen', 'woman']
attribute_sets = [['man', 'king']]

# Evaluate Word2Vec
debiasing_word2vec = MulticlassDebiasing(word2vec_embeddings)
debiasing_word2vec.identify_bias_subspace(defining_sets, min(2, len(defining_sets)))
debiasing_word2vec.remove_bias_components(equality_sets, method='hard')
biased_MAC_word2vec, debiased_MAC_word2vec, p_value_word2vec = debiasing_word2vec.evaluate_debiasing_effect(target_set, attribute_sets)

# Initialize and train GloVe model
glove_model = GloVe(sentences, learning_rate=0.01)
glove_embeddings = {word: glove_model.get_word_vector(word) for word in glove_model.vocab}

# Evaluate GloVe
debiasing_glove = MulticlassDebiasing(glove_embeddings)
debiasing_glove.identify_bias_subspace(defining_sets, min(2, len(defining_sets)))
debiasing_glove.remove_bias_components(equality_sets, method='hard')
biased_MAC_glove, debiased_MAC_glove, p_value_glove = debiasing_glove.evaluate_debiasing_effect(target_set, attribute_sets)

# Print results and further analyze
print("Word2Vec:")
print("Biased MAC:", biased_MAC_word2vec)
print("Debiased MAC:", debiased_MAC_word2vec)
print("P-value:", p_value_word2vec)

print("\nGloVe:")
print("Biased MAC:", biased_MAC_glove)
print("Debiased MAC:", debiased_MAC_glove)
print("P-value:", p_value_glove)
