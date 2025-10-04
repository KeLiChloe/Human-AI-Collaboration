import sqlite3
import pandas as pd
import json
import pandas as pd



def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    
def get_country_by_name_sql(db_path: str, firstname: str, lastname: str) -> str:
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
        
    # Query using the last name first
    query_lastname = """
    SELECT place
    FROM surnames
    WHERE name_lookup = ?
    ORDER BY incidence DESC
    LIMIT 1;
    """
    cursor.execute(query_lastname, (lastname,))
    result = cursor.fetchone()

    # If a result is found with the last name, return it
    if result:
        cursor.close()
        conn.close()
        return result[0]

    # If no result, query using the first name
    query_firstname = """
    SELECT place
    FROM surnames
    WHERE name_lookup = ?
    ORDER BY incidence DESC
    LIMIT 1;
    """
    cursor.execute(query_firstname, (firstname,))
    result = cursor.fetchone()

    # Close the database connection
    cursor.close()
    conn.close()

    # If a result is found with the first name, return it
    if result:
        return result[0]
    
    # If no match for both names, return None
    return None


def create_country_race_diversity_columns(json_file_path, df):
    with open(json_file_path, 'r') as f:
        country_data = json.load(f)

    # Initialize new columns
    df['country_highest_ratio_race'] = None
    df['country_race_shannon_entropy'] = None
    df['country_race_simpson_index'] = None
    df['country_race_inverse_dominance'] = None

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        countries = row['inventors_country']  # Convert string representation of list into actual list
        highest_ratio_race_list = []
        shannon_entropy_list = []
        simpson_index_list = []
        inverse_dominance_list = []
        
        # For each country in the author's list
        for country in countries:
            
            if country == 'England':
                country = 'UNITED KINGDOM'
            if country == 'South Korea':
                country = 'KOREA, REPUBLIC OF'
            if country == 'Vietnam':
                country = 'VIET NAM'
            if country == 'Iran':
                country = 'IRAN, ISLAMIC REPUBLIC OF'
            if country == 'Russia':
                country = 'RUSSIAN FEDERATION'
            country_info = country_data.get(country.upper())  # Match the country in JSON (case insensitive)
            # print(country, country_info)
            if country_info:
                highest_ratio_race_list.append(country_info['highest_ratio_race'])
                shannon_entropy_list.append(country_info['shannon_entropy'])
                simpson_index_list.append(country_info['simpson_index'])
                inverse_dominance_list.append(country_info['inverse_dominance'])
            else:
                break

        if len(highest_ratio_race_list) > 0:
            # Assign the lists as new column values
            df.at[index, 'country_highest_ratio_race'] = highest_ratio_race_list
            df.at[index, 'country_race_shannon_entropy'] = shannon_entropy_list
            df.at[index, 'country_race_simpson_index'] = simpson_index_list
            df.at[index, 'country_race_inverse_dominance'] = inverse_dominance_list
    return df


def process_csv_with_country(input_csv, db_path):

    df = pd.read_csv(input_csv)
    df["inventors_country"] = None

    for idx, row in df.iterrows():

        # 解析名字列表
        firstnames = eval(row["inventors_firstname"]) if isinstance(row["inventors_firstname"], str) else []
        lastnames  = eval(row["inventors_lastname"])  if isinstance(row["inventors_lastname"], str) else []

        countries = []
        for fn, ln in zip(firstnames, lastnames):
            country = get_country_by_name_sql(db_path, fn.lower(), ln.lower())
            countries.append(country)

        # 存结果
        df.at[idx, "inventors_country"] = countries

    # remove rows with None in inventors_country list
    df = df[df["inventors_country"].apply(lambda x: x is not None and all(c is not None for c in x))]
    return df


# 使用示例
if __name__ == "__main__":
    db_path = "database/forebears-surnames.sqlite"  # SQLite 数据库路径
    json_file_path = "meta_data/country_race_diversity_data.json"
    
    input_csv = "code/patent/patent_non-women_results.csv"
    output_csv = "code/patent/patent_non-women_race.csv"
    
    df_tmp = process_csv_with_country(input_csv, db_path)
    df = create_country_race_diversity_columns(json_file_path, df_tmp)
    df.to_csv(f"{output_csv}", index=False)