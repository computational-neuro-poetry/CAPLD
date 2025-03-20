# -*- coding:utf-8 -*-

from glob import glob
import csv
from collections import Counter
import re
import pandas as pd
import numpy as np
from pypinyin import pinyin, Style
from gensim.models import Word2Vec
import os
from scipy.linalg import svd
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']

def generate_dynasty_char_vocab():
    # generate char_vocabulary for each dynasty
    chinese_pattern = re.compile(r'[^\u4e00-\u9fa5]')
    dynasty='Tang Song Yuan Ming Qing Subtlex'.split()
    path = '.\\raw_corpus'
    for dy in dynasty:
        dy_path=path+'\\'+dy+'.csv'
        corpus_dynasty = ""
        # for path in glob(dir_dynasty):
        #     print(path)
        pd_dy = pd.read_csv(dy_path)
        for idx,row in pd_dy.iterrows():
            content = str(row['作者']+row['题目']+row['内容'])
            content_clean = chinese_pattern.sub('', content)
            # print(content_clean)
            corpus_dynasty += content_clean


        char_counter = Counter(corpus_dynasty)

        char_vocab_sorted = char_counter.most_common()


        csv_file ='.\\output\\vocabulary\\char_vocab2_'+dy+'.csv'
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Character', 'Frequency'])
            csv_writer.writerows(char_vocab_sorted)

        print(f"Sorted character vocabulary saved to {csv_file}")


def generate_diachronic_characters_frequency():
    # Define the list of dynasties
    dynasties = ['Tang', 'Song', 'Yuan', 'Ming', 'Qing']

    # Initialize a dictionary to store the vocabulary data for each dynasty
    dynasty_data = {}

    # Loop through each dynasty to read the vocabulary files
    for dy in dynasties:
        file_path = f'.\\output\\vocabulary\\char_vocab_{dy}.csv'  # Assume the file path
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Convert the data to a dictionary with Character as key and Frequency as value
            dynasty_data[dy] = df.set_index('Character')['Frequency'].to_dict()
        else:
            print(f"File not found: {file_path}")
            return

    # Find characters that are common across all dynasties
    common_characters = set(dynasty_data[dynasties[0]].keys())
    for dy in dynasties[1:]:
        common_characters = common_characters.intersection(set(dynasty_data[dy].keys()))

    # Initialize the result list
    result_data = []

    # Loop through each common character to calculate raw frequencies and per million frequencies
    for char in common_characters:
        # Extract raw frequencies for the character across all dynasties
        raw_frequencies = [dynasty_data[dy].get(char, 0) for dy in dynasties]
        # Calculate total frequency across all dynasties
        total_frequency = sum(raw_frequencies)
        # Calculate per million frequencies
        per_million_frequencies = [freq / total_frequency * 1000000 if total_frequency > 0 else 0 for freq in
                                   raw_frequencies]

        # Append the results to the list
        result_data.append([*raw_frequencies, *per_million_frequencies])

    # Create a DataFrame from the result list
    columns = ['Raw_Tang', 'Raw_Song', 'Raw_Yuan', 'Raw_Ming', 'Raw_Qing',
               'PerMillion_Tang', 'PerMillion_Song', 'PerMillion_Yuan', 'PerMillion_Ming', 'PerMillion_Qing']
    result_df = pd.DataFrame(result_data, index=list(common_characters), columns=columns)

    # Save the DataFrame to a CSV file
    output_file = '.\\output\\Diachronic_data\\Diachronic_character_frequencies.csv'
    result_df.to_csv(output_file, index_label='Character')  # Add 'Character' as the index label






def generate_diachronic_diversity():
    # Define the list of dynasties
    dynasties = ['Tang', 'Song', 'Yuan', 'Ming', 'Qing']

    # Initialize a dictionary to store the vocabulary data for each dynasty
    dynasty_vocab = {}

    # Read vocabulary files for each dynasty
    for dy in dynasties:
        vocab_path = fr'.\output\vocabulary\char_vocab_{dy}.csv'
        if os.path.exists(vocab_path):
            vocab_df = pd.read_csv(vocab_path)
            dynasty_vocab[dy] = set(vocab_df['Character'])
        else:
            print(f"Vocabulary file not found: {vocab_path}")
            return

    # Find common characters across all dynasties
    common_characters = set.intersection(*dynasty_vocab.values())

    # Initialize a dictionary to store the diversity counts
    diversity_counts = {char: [0] * len(dynasties) for char in common_characters}

    # Process poetry corpus for each dynasty
    for idx, dy in enumerate(dynasties):
        corpus_path = f'.\\raw_corpus\\{dy}.csv'
        files = glob(corpus_path)

        for file in files:
            if os.path.exists(file):
                poems_df = pd.read_csv(file)
                for _, row in poems_df.iterrows():
                    # Combine title, author, and content into a single text
                    text = str(row['内容'])
                    # Update diversity counts for each common character
                    for char in common_characters:
                        if char in text:
                            diversity_counts[char][idx] += 1
            else:
                print(f"Poetry corpus file not found: {file}")
                return

    # Prepare the result DataFrame
    result_data = []
    for char, counts in diversity_counts.items():
        result_data.append([char] + counts)

    # Create DataFrame with specified columns
    columns = ['Character', 'Tang_diversity', 'Song_diversity', 'Yuan_diversity', 'Ming_diversity', 'Qing_diversity']
    result_df = pd.DataFrame(result_data, columns=columns)

    # Save the result to a CSV file
    output_file = '.\\output\\Diachronic_data\\Diachronic_character_contextual_diversity.csv'
    result_df.to_csv(output_file, index=False)





def get_modern_pinyin_and_tone(char):
    """
    Get modern pinyin and tone for a character.
    - Modern_pinyin: pinyin without tone marks.
    - Modern_tone: '平' for tones 1 and 2; '仄' for tones 3 and 4.
    """
    pinyin_list = pinyin(char, style=Style.TONE2, heteronym=False)  # Get pinyin with tone marks
    if not pinyin_list:
        return "", ""

    pinyin_with_tone = pinyin_list[0][0]

    modern_pinyin = ''.join([c for c in pinyin_with_tone if not c.isdigit()])  # Remove tone numbers
    try:
        tone = int(''.join([c for c in pinyin_with_tone if c.isdigit()]))
        modern_tone = '平' if tone in [1, 2] else '仄'# Extract tone number
    except:
        print(pinyin_with_tone)
        modern_tone =' '

    return modern_pinyin, modern_tone


def load_ancient_tone_and_rhyme(pingshuiyun_file_path):
    """
    Load ancient tone and rhyme data from a local dictionary file.
    """
    ancient_tone_rhyme = {}
    with open(pingshuiyun_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            category, examples = line.split('：')
            examples= re.sub('\[(.*?)]', '', examples)
            tone = category[:2]  # Extract tone (e.g., '上平', '上声')
            rhyme = category  # Extract rhyme (e.g., '二冬')
            for char in examples:
                if char not in ancient_tone_rhyme:
                    ancient_tone_rhyme[char] = []
                ancient_tone_rhyme[char].append((tone, rhyme))
    return ancient_tone_rhyme


def annotate_characters(common_characters, ancient_tone_rhyme):
    """
    Annotate each character with modern and ancient phonetic properties.
    """
    result_data = []
    for char in common_characters:
        modern_pinyin, modern_tone = get_modern_pinyin_and_tone(char)
        ancient_tones = []
        ancient_rhymes = []

        if char in ancient_tone_rhyme:
            for tone, rhyme in ancient_tone_rhyme[char]:
                ancient_tones.append('平' if '平' in tone else '仄')
                ancient_rhymes.append(rhyme)

        # if len(ancient_tones)!=0:
        ancient_tone = ','.join(set(ancient_tones))  # Remove duplicates and join
        ancient_rhyme = ','.join(set(ancient_rhymes))  # Remove duplicates and join
        # else:
        #     print('not found in pingshuiyun')
        #     print(char)
        #     ancient_tone=' '
        #     ancient_rhyme=' '
        result_data.append([char, modern_pinyin, modern_tone, ancient_tone, ancient_rhyme])
    return result_data


def generate_diachronic_character_phonetics():
    # Define the list of dynasties
    dynasties = ['Tang', 'Song', 'Yuan', 'Ming', 'Qing','Subtlex']

    # Initialize a dictionary to store the vocabulary data for each dynasty
    dynasty_vocab = {}

    # Read vocabulary files for each dynasty
    for dy in dynasties:
        vocab_path = fr'.\output\vocabulary\char_vocab_{dy}.csv'
        if os.path.exists(vocab_path):
            vocab_df = pd.read_csv(vocab_path)
            dynasty_vocab[dy] = set(vocab_df['Character'])
        else:
            print(f"Vocabulary file not found: {vocab_path}")
            return

    # Find common characters across all dynasties
    common_characters = set.intersection(*dynasty_vocab.values())

    # Load ancient tone and rhyme data
    ancient_tone_rhyme_file = 'source\pingshuiyun.txt'
    if not os.path.exists(ancient_tone_rhyme_file):
        print(f"Ancient tone and rhyme file not found: {ancient_tone_rhyme_file}")
        return
    ancient_tone_rhyme = load_ancient_tone_and_rhyme(ancient_tone_rhyme_file)

    # Annotate common characters
    result_data = annotate_characters(common_characters, ancient_tone_rhyme)

    # Create DataFrame and save to CSV
    columns = ['Character', 'Modern_pinyin', 'Modern_tone', 'Ancient_tone', 'Ancient_rhyme']
    result_df = pd.DataFrame(result_data, columns=columns)
    output_file = '.\\output\\Diachronic_data\\Diachronic_characters_phonetics.csv'
    result_df.to_csv(output_file, index=False)
    print(f"Character annotations have been saved to {output_file}")


def train_dynasty_word2vec():
    chinese_pattern = re.compile(r'[a-zA-Z,.?()（）【】\[\]!，\'" ]+')
    dynasties = ['Tang', 'Song', 'Yuan', 'Ming', 'Qing']

    # Ensure the output directory exists
    output_dir = './output/Dynasty_embeddings/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for dy in dynasties:
        print(f"Training Word2Vec model for {dy} dynasty...")
        corpus_dynasty = []  # Initialize corpus for the current dynasty

        # Load poetry files for the current dynasty
        dir_dynasty = fr".\raw_corpus\{dy}.csv"
        # for path in glob(dir_dynasty):
        print(f"Processing file: {dir_dynasty}")
        pd_dy = pd.read_csv(dir_dynasty)
        for idx, row in pd_dy.iterrows():
            # Combine title, author, and content
            content =str(row["题目"])  + str(row["作者"]) + str(row["内容"])
            # Remove non-Chinese characters
            content_clean = chinese_pattern.sub('', content)
            # Split into sentences and convert to lists of characters
            content_clean = re.split('。|？|！|，', content_clean)
            content_clean = [list(i) for i in content_clean if i]
            corpus_dynasty.extend(content_clean)

        # Train Word2Vec model for the current dynasty
        if corpus_dynasty:
            model = Word2Vec(sentences=corpus_dynasty[:500], vector_size=300, window=5, min_count=5, sg=1, workers=4)
            model.train(corpus_dynasty, total_examples=model.corpus_count, epochs=10)

            # Save the model
            model_path = os.path.join(output_dir, f'{dy}_word2vec.model')
            model.save(model_path)
            print(f"Word2Vec model for {dy} dynasty saved to {model_path}")


            embeddings_path = os.path.join(output_dir, f'{dy}_word2vec.txt')
            with open(embeddings_path, 'a', encoding='utf-8') as f:
                for word in model.wv.index_to_key:
                    vector = model.wv[word]
                    f.write(f"{word} {' '.join(map(str, vector))}\n")
            print(f"Word embeddings for {dy} dynasty saved to {embeddings_path}")
        else:
            print(f"No data for {dy} dynasty. Skipping model training.")



def generate_co_occurrence_matrix():
    chinese_pattern = re.compile(r'[^\u4e00-\u9fa5]')
    dynasties = [ 'Tang', 'Song', 'Yuan', 'Ming','Qing']
    for dy in dynasties:

        vocab_df = pd.read_csv(fr'.\output\vocabulary\char_vocab_{dy}.csv')

        filtered_vocab_df = vocab_df[vocab_df["Frequency"] > 10]


        characters = filtered_vocab_df["Character"].tolist()


        co_occurrence_matrix = pd.DataFrame(
            data=np.zeros((len(characters), len(characters)), dtype=int),
            index=characters,
            columns=characters
        )



        dir_path = fr'.\raw_corpus\{dy}.csv'
        # for path in dir_path:
        each_dy_poems = pd.read_csv(dir_path)
        for _, row in each_dy_poems.iterrows():
            text = str(row["题目"])  + str(row["作者"]) + str(row["内容"])
            for i in range(len(text) - 1):
                first_char = text[i]
                second_char = text[i + 1]

                if first_char in characters and second_char in characters:
                    co_occurrence_matrix.loc[first_char, second_char] += 1


        co_occurrence_matrix.to_csv(fr'.\output\bigram_cooccurrence_matrices\co_occurrence_{dy}.csv')

if __name__=="__main__":
    generate_dynasty_char_vocab()
    generate_diachronic_diversity()
    generate_diachronic_characters_frequency()
    generate_diachronic_character_phonetics()
    generate_co_occurrence_matrix()
    train_dynasty_word2vec()




