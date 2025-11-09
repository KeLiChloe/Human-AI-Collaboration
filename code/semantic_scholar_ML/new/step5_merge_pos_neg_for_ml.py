import sys
import pandas as pd

def concat_csv(file1, file2, output_file):
    # Read both CSVs
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Check if they have the same columns
    if set(df1.columns) != set(df2.columns):
        print("Error: The two CSV files do not have the same columns.")
        print("File1 columns:", df1.columns.tolist())
        print("File2 columns:", df2.columns.tolist())
        sys.exit(1)

    # Reorder columns in df2 to match df1 (just in case)
    df2 = df2[df1.columns]

    # Concatenate
    result = pd.concat([df1, df2], ignore_index=True)

    
    # select only a few columns to save to csv
    columns_to_save = ['title', 'paper_abstract', 'count_inequality_words',
                       'female_mean', 'female_max', 'female_min','first_author_female_score',
                       'natural_sciences', 'engineering_and_technology', 'social_sciences',
                       'country_race_shannon_entropy_mean', 'paper_race_shannon_entropy',
                        'percentage_of_asian', 'percentage_of_white', 'percentage_of_black', 'percentage_of_hispanic',
                        'acad_ineq_t-0', 'acad_ineq_t-1', 'acad_ineq_t-2', 'acad_ineq_t-3', 'acad_ineq_3yr_avg', 
                        'news_ineq_t-0', 'news_ineq_t-1', 'news_ineq_t-2', 'news_ineq_t-3', 'news_ineq_3yr_avg', 
                        'news_gender_ineq_t-0', 'news_gender_ineq_t-1', 'news_gender_ineq_t-2', 'news_gender_ineq_t-3', 'news_gender_ineq_3yr_avg', 
                        'news_econ_ineq_t-0', 'news_econ_ineq_t-1', 'news_econ_ineq_t-2', 'news_econ_ineq_t-3', 'news_econ_ineq_3yr_avg', 
                        'news_race_ineq_t-0', 'news_race_ineq_t-1', 'news_race_ineq_t-2', 'news_race_ineq_t-3', 'news_race_ineq_3yr_avg']
    
    result = result[columns_to_save]
    

    # Save to output file
    result.to_csv(output_file, index=False)
    print(f"✅ Concatenated file saved as {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python concat_csv.py <file1.csv> <file2.csv> <output.csv>")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]
    output_file = sys.argv[3]

    concat_csv(file1, file2, output_file)
