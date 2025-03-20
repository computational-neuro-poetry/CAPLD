import pandas as pd
import math
import re
import glob
from collections import defaultdict
import os
import numpy as np



def calculate_entropy(frequencies):
    # Function to calculate entropy given a dictionary of frequencies
    total = sum(frequencies.values())
    entropy = -sum((freq / total) * math.log2(freq / total) for freq in frequencies.values() if freq > 0)
    return entropy

def statistic_L_R_Entropy():
    # Function to calculate left and right entropies for each dynasty and combine them intoCSV files
    chinese_pattern = re.compile(r'[^\u4e00-\u9fa5]')
    dynasties = ['Tang', 'Song', 'Yuan', 'Ming', 'Qing', 'Subtlex']

    for dy in dynasties:
        vocab_df = pd.read_csv(fr'.\\output\vocabulary\char_vocab_{dy}.csv')
        characters = vocab_df["Character"].tolist()
        frequencies = vocab_df["Frequency"].tolist()


        total_frequency = sum(frequencies)
        normalized_frequencies = [freq / total_frequency for freq in frequencies]


        left_neighbors = defaultdict(lambda: defaultdict(int))
        right_neighbors = defaultdict(lambda: defaultdict(int))



        dir_path = fr'.\raw_corpus\{dy}.csv'

        each_dy_poems = pd.read_csv(dir_path)
        for _, row in each_dy_poems.iterrows():
            text = str(row["内容"])
            text = chinese_pattern.sub('', text)  # 去除非中文字符

            # 统计左右邻接字的频率
            for i in range(1, len(text) - 1):
                char = text[i]
                if char in characters:
                    left_neighbors[char][text[i - 1]] += 1
                    right_neighbors[char][text[i + 1]] += 1




        entropy_data = []
        for i, char in enumerate(characters):
            left_entropy = calculate_entropy(left_neighbors[char])
            right_entropy = calculate_entropy(right_neighbors[char])
            entropy_data.append((char, frequencies[i], normalized_frequencies[i], left_entropy, right_entropy))


        entropy_df = pd.DataFrame(entropy_data, columns=['Character', 'Frequency', 'Normalize', 'Left_Entropy', 'Right_Entropy'])
        entropy_df.to_csv(fr'.\output\entropies\char_entropy_{dy}.csv', index=False)


def generate_entropy():
    # List of dynasties
    dynasties = ['Tang', 'Song', 'Yuan', 'Ming', 'Qing']

    # Initialize an empty DataFrame to store the combined entropy data
    combined_data = None

    # Dictionary to store DataFrames for each dynasty
    dynasty_data = {}

    # Load entropy data for each dynasty
    for dy in dynasties:
        file_path = fr'.\output\entropies\char_entropy_{dy}.csv'
        if os.path.exists(file_path):
            dynasty_data[dy] = pd.read_csv(file_path)
        else:
            print(f"File not found: {file_path}")
            return

    # Find common characters across all dynasties
    common_characters = set(dynasty_data[dynasties[0]]['Character'])
    for dy in dynasties[1:]:
        common_characters = common_characters.intersection(set(dynasty_data[dy]['Character']))

    # Initialize the combined DataFrame with common characters
    combined_data = pd.DataFrame({'Character': list(common_characters)})

    # Extract and merge entropy values for each dynasty
    for dy in dynasties:
        left_entropy_col = f'{dy}Left_Entropy'
        right_entropy_col = f'{dy}Right_Entropy'

        # Merge left entropy
        combined_data = combined_data.merge(
            dynasty_data[dy][['Character', 'Left_Entropy']].rename(columns={'Left_Entropy': left_entropy_col}),
            on='Character', how='left'
        )

        # Merge right entropy
        combined_data = combined_data.merge(
            dynasty_data[dy][['Character', 'Right_Entropy']].rename(columns={'Right_Entropy': right_entropy_col}),
            on='Character', how='left'
        )

    # Save the combined data to a single CSV file
    output_file = fr'.\output\time_series\Diachronic_character_entropies.csv'
    combined_data.to_csv(output_file, index=False)
    print(f"Combined entropy data for common characters saved to {output_file}")

def generate_frequency_diversity():
    # Define the list of dynasties
    dynasties = ['Tang', 'Song', 'Yuan', 'Ming', 'Qing']

    # Load the frequency data
    input_file = fr'.\output\Diachronic_data\Diachronic_character_frequencies.csv'
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return

    df = pd.read_csv(input_file)

    # Initialize columns for frequency diversity and log frequency diversity
    for dy in dynasties[1:]:  # Skip Tang dynasty as it is the reference
        df[f'{dy}_fd'] = df[f'PerMillion_{dy}'] / df['PerMillion_Tang']
        df[f'{dy}_log_fd'] = np.log(df[f'{dy}_fd'])

    # Select and reorder the columns as specified
    output_columns = ['Character'] + [f'{dy}_fd' for dy in dynasties[1:]] + [f'{dy}_log_fd' for dy in dynasties[1:]]
    df = df[output_columns]

    # Ensure the output directory exists
    output_dir = fr'.\output\time_series'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the results to a new CSV file
    output_file = os.path.join(output_dir, 'Diachronic_character_frequency_diversity.csv')
    df.to_csv(output_file, index=False)
    print(f"Frequency diversity data has been saved to {output_file}")


if __name__=="__main__":
    # statistic_L_R_Entropy()
    # generate_entropy()
    generate_frequency_diversity()