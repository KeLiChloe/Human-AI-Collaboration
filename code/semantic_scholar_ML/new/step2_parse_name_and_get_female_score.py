import pandas as pd
import requests
from requests.auth import HTTPBasicAuth  # Only if Solr requires authentication
from tqdm import tqdm
import sys
import os
import csv
import pandas as pd
import re
import json

# This script did the following:
# 1. parse names to first and last names
# 2. get female score
# 3. map fields of study to three categories
# 4. count the number of inequality-related words in title and abstract   

solr_url = 'https://solr-facebooknames.totosearch.org/solr/facebook_lastname/select'
username = 'research'
password = 'insead123456'

def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    
# Return three values for a name: 
# 1. the prob of being last name, 
# 2. the count of as last name in the database, 
# 3. the prob of being female
def get_query_from_facebook(name, solr_url, username=None, password=None):
    params = {
        'q': f'lastname:"{name}"',
        'wt': 'json',  # Requesting JSON response
        "sort":"lastname_count desc",
        "rows":"1"
    }
    # Make the HTTP GET request to Solr
    response = requests.get(solr_url, params=params, auth=HTTPBasicAuth(username, password) if username and password else None)
    if response.status_code == 200:
        # Parse the response JSON
        data = response.json()
        if 'response' in data and 'docs' in data['response'] and len(data['response']['docs']) > 0:
            result = data['response']['docs'][0]
            return result['lastname_percent']/100,  result['lastname_count'], result['female_percent']/100
        else:
            return 0, 0, 0.5  # Name not found
        
    else:
        raise ValueError(f"Failed to query Solr for name '{name}'. Status code:", response.status_code)


# return order: first name, last name, female_percent
def determine_first_last_female_country_facebook(name, solr_url, username=None, password=None):
    
    name_parts = name.strip(" ").split()
    first_word = name_parts[0].replace("\\", "")  # remove "\" in the string
    last_word = name_parts[-1].replace("\\", "")  # remove "\" in the string
    
    # determining which are first and last names
    first_word_percent, first_word_count, female_percent_first_word  = get_query_from_facebook(first_word, solr_url, username, password)
    last_word_percent, last_word_count, female_percent_last_word = get_query_from_facebook(last_word, solr_url, username, password)
    
    if len(name_parts) == 1:
        return name_parts[0], name_parts[0], female_percent_first_word # Treat as both first and last name
    
    if '-' in first_word:
        return first_word, last_word, female_percent_first_word  # First word as first name
    elif '-' in last_word:
        return last_word, first_word, female_percent_last_word  # Last word as first name
    
    if '.' in first_word:
        return first_word, last_word, female_percent_first_word # First word as first name
    elif '.' in last_word:
        return last_word, first_word, female_percent_last_word # Last word as first name
    
    if len(first_word)==1:
        return first_word, last_word, female_percent_first_word # First word as first name
    elif len(last_word)==1:
        return last_word, first_word, female_percent_last_word  # Last word as first name
    

    # Decide which percent to trust based on the higher lastname_count
    # Under this logic, if both parts of the names cannot be queried, always treat first word as first name

    if first_word_count >= last_word_count:
        
        # Trust first word's percent
        if first_word_percent < 0.5:
            return first_word, last_word, female_percent_first_word # First word as first name
        else:
            return last_word, first_word, female_percent_last_word   # last word is first name
            
    else:
        # Trust last word's percent
        if last_word_percent >= 0.5:
            return first_word, last_word, female_percent_first_word  # First word as first name
        else:
            return last_word, first_word, female_percent_last_word  ## last word is first name

def get_processed_row_count(output_file):
    if os.path.exists(output_file):
        processed_rows = len(pd.read_csv(output_file))  
    else:
        processed_rows = 0
    return processed_rows

field_groups = {
    'natural_sciences': [
        'biology', 'chemistry', 'physics', 'geology', 'geography',
        'environmental science', 'medicine', 'materials science'
    ],
    'engineering_and_technology': [
        'engineering', 'computer Science', 'mathematics'
    ],
    'social_sciences': [
        'sociology', 'political science', 'psychology', 
        'economics', 'history', 'art', 'philosophy'
    ]
}

def map_fields_of_study(fields_list, field_groups):
    if isinstance(fields_list, str):
        fields_list = eval(fields_list.lower())
    mapped_fields = {key: 0 for key in field_groups.keys()}
    for field in fields_list:
        for category, fields in field_groups.items():
            if field in fields:
                mapped_fields[category] = 1
    return mapped_fields    

# Function to count word occurrences in a given text
def count_words(text, words):
    text = text.lower()
    total_count = 0
    for word in words:
        # Escape special characters and count occurrences using regular expressions
        total_count += len(re.findall(r'\b' + re.escape(word) + r'\b', text))
    
    return total_count



if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python step2_parse_name_and_get_female_score.py <input_file_path> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    df = pd.read_csv(input_file)
    wordlist = read_json_file('meta_data/wordlist.json')
    
    inequality_words = []
    for category in ['race', 'gender', 'economic']:
        inequality_words += wordlist[category]["0"] + wordlist[category]["1"] + wordlist[category]["-1"]

    # 如果已有部分输出，就读取现有行数，跳过
    processed_rows = get_processed_row_count(output_file)
    mode = "a" if processed_rows > 0 else "w"
    print(f"Starting from row {processed_rows}")

    DATA_FIELDS = ["title", "paper_abstract", "year", "fields_of_study",
            "firstname", "lastname", 
            "female_mean", "female_max", "female_min", "first_author_female_score",
            "count_inequality_words"
            ] + list(field_groups.keys())

    with open(output_file, mode, newline='', encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=DATA_FIELDS)
        if processed_rows == 0:
            writer.writeheader()

        for idx, row in tqdm(df.iloc[processed_rows:].iterrows(), total=len(df)-processed_rows, desc="Processing rows"):
            fields_of_study = eval(row["fields_of_study"].lower())
            mapped_fields = map_fields_of_study(fields_of_study, field_groups)
            
            author_list = eval(row["authors"].lower()) 
            
            firstnames, lastnames, female_percents = [], [], []
            
            for author in author_list:
                firstname, lastname, female_percent = determine_first_last_female_country_facebook(
                    author, solr_url, username, password
                )
                firstnames.append(firstname)
                lastnames.append(lastname)
                female_percents.append(female_percent)

            female_mean = sum(female_percents)/len(female_percents) if female_percents else None
            female_max = max(female_percents) if female_percents else None
            female_min = min(female_percents) if female_percents else None
            first_author_score = female_percents[0] if female_percents else None

            writer.writerow({
                "title": row.get("title", ""),
                "paper_abstract": row.get("paper_abstract", ""),
                "year": row.get("year", ""),
                "fields_of_study": row.get("fields_of_study", ""),
                
                "firstname": firstnames,
                "lastname": lastnames,
                
                "female_mean": female_mean,
                "female_max": female_max,
                "female_min": female_min,
                "first_author_female_score": first_author_score,
                "count_inequality_words": count_words(str(row['title']) + ' ' + str(row['paper_abstract']), inequality_words),
                **mapped_fields,
            })
            
    
