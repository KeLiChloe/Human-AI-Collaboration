import sqlite3
import json
import math

def get_highest_ratio_race(row):
    # Extract the relevant race columns from the row
    race_columns = ['white', 'black', 'asian', 'hispanic', 'native_hawaiian_or_other_pacific_islander',
                    'native_americans', 'mixed', 'other']
    
    # Get the race with the highest percentage
    highest_race = max(race_columns, key=lambda race: row.get(race, 0))
    return highest_race

# Function to calculate the Shannon-Weiner Index (Entropy)
def calculate_shannon_entropy(proportions):
    entropy = -sum([p * math.log(p) for p in proportions if p > 0])
    return entropy

# Function to calculate Simpson's Diversity Index (actually 1 - D)
def calculate_simpson_index(proportions):
    simpson_index = 1 - sum([p ** 2 for p in proportions])
    return simpson_index

# Function to calculate the Inverse of Dominance
def calculate_inverse_dominance(proportions):
    inverse_dominance = 1 / max(proportions)
    return inverse_dominance


# Helper function to calculate all metrics for a country
def calculate_diversity_metrics(row):
    race_columns = ['white', 'black', 'asian', 'hispanic', 'native_hawaiian_or_other_pacific_islander',
                    'native_americans', 'mixed', 'other']
    
    # Get the race proportions from the row, normalized to percentages (e.g., 0.50 for 50%)
    proportions = [row.get(race, 0) / 100 for race in race_columns if row.get(race, 0) > 0]
    
    if not proportions:
        return {
            "shannon_entropy": 0,
            "simpson_index": 0,
            "fractionalization_index": 0,
            "inverse_dominance": 0
        }

    # Calculate each diversity metric
    shannon_entropy = calculate_shannon_entropy(proportions)
    simpson_index = calculate_simpson_index(proportions)
    inverse_dominance = calculate_inverse_dominance(proportions)

    return {
        "shannon_entropy": shannon_entropy,
        "simpson_index": simpson_index,
        "inverse_dominance": inverse_dominance
    }

# Main function to create the JSON file with diversity metrics for each country
def create_country_race_json(db_path, output_json):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query the races table
    query = "SELECT * FROM races"
    cursor.execute(query)

    # Fetch all results
    rows = cursor.fetchall()

    # Get column names from the table
    columns = [description[0] for description in cursor.description]

    # Create a dictionary to store the result
    country_dict = {}

    for row in rows:
        row_dict = dict(zip(columns, row))  # Convert row to dictionary
        
        # Get the country name
        country = row_dict['country']
        
        # Determine the highest ratio race
        highest_ratio_race = get_highest_ratio_race(row_dict)

        # Calculate diversity metrics
        diversity_metrics = calculate_diversity_metrics(row_dict)
        
        # Populate the dictionary for the country
        country_dict[country] = {
            "highest_ratio_race": highest_ratio_race,
            **diversity_metrics  # Include all diversity metrics
        }
    
    # Save the dictionary as a JSON file
    with open(output_json, 'w') as json_file:
        json.dump(country_dict, json_file, indent=4)
    
    # Close the database connection
    cursor.close()
    conn.close()

# Example usage
db_path = '/Users/like/Desktop/Research/Human-AI/database/race-by-country.sqlite'  # Replace with the actual path
output_json = '/Users/like/Desktop/Research/Human-AI/meta_data/country_race_diversity_data.json'

create_country_race_json(db_path, output_json)
