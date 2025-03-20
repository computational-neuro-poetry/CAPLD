
from glob import glob

import pandas as pd
import numpy as np

from gensim.models import Word2Vec
from scipy.linalg import svd
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
import random
random.seed(200)
import os




def spearman_correlation(dy1, dy2):
    """
    Calculate the Spearman rank correlation coefficient between human scores and semantic similarities.
    """
    # Extract human scores
    score_dict = {}

    paths = glob(fr'source\human_score\*.xlsx')
    paths = [file for file in paths if not os.path.basename(file).startswith("~$")]
    for path in paths:
        xlsx_df = pd.read_excel(path,engine='openpyxl')
        for idx, row in xlsx_df.iterrows():
            if len(row[0]) == 1:  # Ensure it's a single character
                char = row[0]
                score = row[1]
                if char in score_dict.keys():
                    score_dict[char].append(score)
                else:
                    score_dict[char] = []
                    score_dict[char].append(score)
    score_dict = {k: np.mean(v) for k, v in score_dict.items()}  # Calculate the average score for each character

    # Load word2vec models
    path1 = '.\\output\\Dynasty_embeddings\\' + dy1 + '_word2vec.model'
    path2 = '.\\output\\Dynasty_embeddings\\' + dy2 + '_word2vec.model'

    dy1_model = Word2Vec.load(path1)
    dy2_model = Word2Vec.load(path2)

    # Get the common vocabulary between the two models
    common_words = list(set(dy1_model.wv.index_to_key) & set(dy2_model.wv.index_to_key))

    # Check if anchor words overlap with scored words (comment out to allow overlap, do not remove overlapping words)
    common_words = [i for i in common_words if i not in score_dict.keys()]

    # Build the word vector matrices for the common vocabulary
    dy1_matrix = np.array([dy1_model.wv[word] for word in common_words])
    dy2_matrix = np.array([dy2_model.wv[word] for word in common_words])

    # Align the two vector spaces
    U, _, Vt = svd(dy2_matrix.T @ dy1_matrix)
    # Compute the rotation matrix
    R = U @ Vt

    # Extract words from the list
    words = list(score_dict.keys())

    # Calculate semantic similarities
    embedding_similarity = {}
    for word in words:
        if word in dy1_model.wv and word in dy2_model.wv:
            dy1_vec = dy1_model.wv[word]
            aligned_dy2_vec = dy2_model.wv[word] @ R
            similarity = np.dot(dy1_vec, aligned_dy2_vec) / (np.linalg.norm(dy1_vec) * np.linalg.norm(aligned_dy2_vec))
            embedding_similarity[word] = similarity

    # Extract human scores and semantic similarities
    human_scores = []
    machine_similarities = []
    for word in score_dict:
        if word in embedding_similarity:
            human_scores.append(score_dict[word])
            machine_similarities.append(embedding_similarity[word])

    # Calculate the Spearman rank correlation coefficient
    correlation, p_value = spearmanr(human_scores, machine_similarities)

    # Output results
    print(score_dict)
    print(embedding_similarity)
    print(f"Spearman correlation: {correlation}")
    print(f"P-value: {p_value}")
    return embedding_similarity, score_dict, correlation, p_value


if __name__ == "__main__":
    for i in ['Song', 'Yuan', 'Ming', 'Qing']:
        spearman_correlation(i, 'Tang')