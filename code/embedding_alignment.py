import re
import numpy as np
from scipy.linalg import svd, orthogonal_procrustes
import glob
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

def align_embeddings_without_anchors(matrix_a, matrix_b):
    """
    Align two embedding spaces without using anchor words
    :param matrix_a: The word embedding matrix of Model A (number of words x dimension)
    :param matrix_b: The word embedding matrix of Model B (number of words x dimension)
    :return: The aligned embedding matrix of Model B
    """
    # Perform Singular Value Decomposition on the embeddings of A and B
    U, _, Vt = svd(matrix_a.T @ matrix_b)

    # Compute the optimal rotation matrix
    rotation_matrix = U @ Vt

    # Apply the rotation matrix to all word embeddings in Model B
    aligned_matrix_b = matrix_b @ rotation_matrix

    return aligned_matrix_b, rotation_matrix


def train_dynasty_word2vec():
    chinese_pattern = re.compile(r'[a-zA-Z,.?()（）【】\[\]!，\'" ]+')
    dynasty = ['Tang', 'Song', 'Yuan', 'Ming', 'Qing', 'subtlex']
    corpus_dynasty = []  # Overall ancient poetry corpus
    for dy in dynasty:
        dir_dynasty = fr".\poem_corpus\{dy}\*.csv"
        for path in glob.glob(dir_dynasty):
            print(path)
            pd_dy = pd.read_csv(path)
            for idx, row in pd_dy.iterrows():
                content = str(row[0]) + str(row[2]) + str(row[3])
                content_clean = chinese_pattern.sub('', content)  # Remove non-Chinese characters
                content_clean = re.split('。|？|！', content_clean)
                content_clean = [list(i) for i in content_clean if i]
                corpus_dynasty.extend(content_clean)

    # Train the Dynasty Word2Vec model
    model = Word2Vec(sentences=corpus_dynasty, vector_size=300, window=5, min_count=5, sg=1, workers=4)  # sg=0 Training algorithm: 1 for skip-gram; otherwise CBOW.
    # Build vocabulary
    model.build_vocab(corpus_dynasty)

    # Train the model
    model.train(corpus_dynasty, total_examples=model.corpus_count, epochs=10)

    # Save the model
    model_path = fr'./output/Diachronic_embeddings/dynasty_word2vec.model'
    model.save(model_path)
    print(f"Word2Vec model saved to {model_path}")


def align_embedding_space():
    def align_embeddings(reference_matrix, target_matrix):
        """
        Align the target embedding matrix to the reference embedding matrix
        :param reference_matrix: The reference embedding matrix (number of words x dimension)
        :param target_matrix: The target embedding matrix (number of words x dimension)
        :return: The aligned target embedding matrix
        """
        R, _ = orthogonal_procrustes(target_matrix, reference_matrix)
        aligned_matrix = target_matrix @ R
        return aligned_matrix

    def align_all_embeddings(models):
        """
        Align multiple embedding matrices to the same reference space
        :param models: A list of Word2Vec models
        :return: The list of aligned embedding matrices and the common vocabulary
        """
        # Get the common vocabulary of all models
        common_vocab = set(models[0].wv.index_to_key)
        for model in models[1:]:
            common_vocab.intersection_update(model.wv.index_to_key)

        common_vocab = sorted(list(common_vocab))  # Sort for consistency

        # Build the embedding matrices for the common vocabulary
        embedding_matrices = []
        for model in models:
            matrix = np.array([model.wv[word] for word in common_vocab])
            embedding_matrices.append(matrix)

        # Use the first embedding matrix as the reference
        reference_matrix = embedding_matrices[0]

        # Align all other embedding matrices to the reference space
        aligned_matrices = [reference_matrix]
        for matrix in embedding_matrices[1:]:
            aligned_matrix = align_embeddings(reference_matrix, matrix)
            aligned_matrices.append(aligned_matrix)

        return aligned_matrices, common_vocab

    dynasty = ['Tang', 'Song', 'Yuan', 'Ming', 'Qing']
    # Example: Load 6 Word2Vec models and extract the embedding matrices
    model_paths = [fr'./output/Dynasty_embeddings/{dy}_word2vec.model' for dy in dynasty]

    models = [Word2Vec.load(path) for path in model_paths]

    # Align all embedding matrices
    aligned_matrices, common_vocab = align_all_embeddings(models)

    # Save the aligned word vectors for each model
    for i, dy in enumerate(dynasty):
        model = models[i]
        aligned_matrix = aligned_matrices[i]
        with open(fr'./output/Dynasty_embeddings/{dy}_word2vec.txt', 'w', encoding='utf-8') as f:
            f.write(f"{len(common_vocab)} {aligned_matrix.shape[1]}\n")
            for j, word in enumerate(common_vocab):
                vector = ' '.join(map(str, aligned_matrix[j]))
                f.write(f"{word} {vector}\n")

    return aligned_matrices, common_vocab, models  # Output the aligned embedding spaces



def generate_dynasty_cosine_similarity():
    aligned_matrices,common_vocab,models=align_embedding_space()
    # Get the first matrix as the base matrix
    base_matrix = aligned_matrices[0]
    other_matrices = aligned_matrices[1:]

    # Initialize the result list
    results = []

    # Iterate through each word in the common vocabulary
    for idx, word in enumerate(common_vocab):
        # Get the vector of the current word in the base matrix
        base_vector = base_matrix[idx].reshape(1, -1)

        # Calculate the cosine similarity with the corresponding word vectors in other matrices
        similarities = []
        for matrix in other_matrices:
            other_vector = matrix[idx].reshape(1, -1)
            similarity = cosine_similarity(base_vector, other_vector)[0][0]
            similarities.append(similarity)

        # Append the result to the list
        results.append([word] + similarities)

    # Create a DataFrame
    columns = ['Character', 'Tang&Song', 'Tang&Yuan', 'Tang&Ming', 'Tang&Qing', 'Tang&Subtlex']
    df = pd.DataFrame(results, columns=columns)

    # Save to a CSV file
    df.to_csv(r'.\output\time_series\Diachronic_character_similarities.csv', index=False)
    print("Cosine similarity results have been saved to 'Diachronic_character_similarities.csv'.")

if __name__=="__main__":
    generate_dynasty_cosine_similarity()