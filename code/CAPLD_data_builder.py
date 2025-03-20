import os
import pandas as pd
import re
import math
from collections import defaultdict


# Function to clean text, removing special characters and punctuation
def clean_text(text):
    # Use regular expression to keep only Chinese characters
    cleaned_text = re.sub(r'[^\u4e00-\u9fff]', '', text)
    return cleaned_text

# Function to get the stroke count of a Chinese character
def get_stroke_count():
    """
    Retrieve the stroke count of a given Chinese character from a specified text file.

    Parameters:
        word (str): The Chinese character to query.
        filename (str): The path to the text file containing the character and stroke count information.

    Returns:
        int: The stroke count of the character if found; otherwise, -1.
    """
    stroke_file = fr'source\chinese_unicode_table.txt'
    stroke_file=open(stroke_file, 'r', encoding='utf-8')
    lines=stroke_file.readlines()
        # with open(stroke_file, 'r', encoding='utf-8') as file:
        #     lines = file.readlines()

        # Process each line starting from the 7th line (index 6)
    stroke_dict={}
    for line in lines[6:]:
        columns = line.strip().split(' ')
        columns=[c for c in columns if len(c)!=0]# Split columns by tab
        if len(columns) >= 7:  # Ensure there are at least 7 columns
            character = columns[0]  # The first column is the character
            stroke_count = columns[6]  # The 7th column is the stroke count
            stroke_dict[character]=int(stroke_count)

    return stroke_dict


# Initialize dictionaries to store character frequency and environmental diversity
char_frequency = defaultdict(int)
char_environment_diversity = defaultdict(set)
char_tone_rhyme = defaultdict(lambda: {'Tone': [], 'Rhyme': []})  # Store Tone and Rhyme for each character

# Path to the folder containing CSV files
folder_path = '.\\raw_corpus'  # Replace with your folder path

# Load the pingshuiyun.txt file and parse its content
pingshuiyun_path = 'source\pingshuiyun.txt'  # Path to the pingshuiyun.txt file
file=open(pingshuiyun_path, 'r', encoding='utf-8')
pingshuiyun_lines=file.readlines()
# with open(pingshuiyun_path, 'r', encoding='utf-8') as file:
#     pingshuiyun_lines = file.readlines()

# Process each line in pingshuiyun.txt to build a mapping of characters to their Tone and Rhyme
for line in pingshuiyun_lines:
    line = line.strip()
    if line:  # Ensure the line is not empty
        parts = line.split('：')  # Split by colon
        category = parts[0]  # Category name (e.g., "上平二冬")
        examples = parts[1]  # Characters in this category

        # Determine Tone based on the category name
        if '上平' in category or '下平' in category:
            tone = '平'
        elif '上声' in category or '去声' in category or '入声' in category:
            tone = '仄'
        else:
            tone = 'Unknown'  # In case of unexpected category

        rhyme = category  # Use the entire category name as Rhyme

        # Assign Tone and Rhyme to each character in the category
        for char in examples:
            if tone not in char_tone_rhyme[char]['Tone']:
                char_tone_rhyme[char]['Tone'].append(tone)
            if rhyme not in char_tone_rhyme[char]['Rhyme']:
                char_tone_rhyme[char]['Rhyme'].append(rhyme)

# Iterate through all CSV files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        # Read the CSV file
        df = pd.read_csv(file_path, encoding='utf-8')

        # Iterate through each row in the content column
        for content in df['内容']:
            cleaned_content = clean_text(content)
            # Tokenize by character
            for char in cleaned_content:
                char_frequency[char] += 1
                # Add the current poem's title to the environmental diversity set for the character
                char_environment_diversity[char].add(content)

# Calculate the total number of characters for frequency_per_million
total_characters = sum(char_frequency.values())
stroke_dict=get_stroke_count()
# Convert results to a DataFrame
result_data = {
    'Character': [],
    'Frequency': [],
    'Log_frequency': [],
    'Frequency_per_million': [],
    'CD': [],
    'Log_cd': [],
    'Strokes': [],
    'Tone': [],
    'Rhyme': []
}

for char, freq in char_frequency.items():
    result_data['Character'].append(char)
    result_data['Frequency'].append(freq)
    result_data['Log_frequency'].append(math.log(freq) if freq > 0 else 0)
    result_data['Frequency_per_million'].append((freq / total_characters) * 1e6)
    result_data['CD'].append(len(char_environment_diversity[char]))
    result_data['Log_cd'].append(math.log(len(char_environment_diversity[char])) if len(char_environment_diversity[char]) > 0 else 0)
    result_data['Strokes'].append(stroke_dict[char] if stroke_dict[char] else -1)
    result_data['Tone'].append('/'.join(char_tone_rhyme[char]['Tone']))
    result_data['Rhyme'].append(' '.join(char_tone_rhyme[char]['Rhyme']))

result_df = pd.DataFrame(result_data)

# Save the result to a CSV file
result_df.to_csv(r'.\output\CAPLD.csv', index=False, encoding='utf-8-sig')

print("The CAPLD corpus with Tone, Rhyme, Stroke_count, and Frequency_per_million has been generated and saved to 'char_corpus_with_tone_rhyme.csv'.")