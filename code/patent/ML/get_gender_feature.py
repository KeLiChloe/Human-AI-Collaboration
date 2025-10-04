
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth  # Only if Solr requires authentication
import sys


solr_url = 'https://solr-facebooknames.totosearch.org/solr/facebook_lastname/select'
username = 'research'
password = 'insead123456'

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
            return result['female_percent']/100
        else:
            return 0.5 # Name not found
        
    else:
        print(f"Failed to query Solr for name '{name}'. Status code:", response.status_code)
        return None

def process_csv(input_csv, output_csv, solr_url, username=None, password=None, batch_size=1000):
    # 如果已有输出文件就继续处理，否则从输入文件开始
    try:
        df = pd.read_csv(output_csv)
        print(f"继续处理已有文件 {output_csv}")
    except FileNotFoundError:
        df = pd.read_csv(input_csv)
        df["female_percents"] = None
        df["mean_female_percent"] = None
        print(f"新建文件 {output_csv}")

    processed = 0

    for idx, row in df.iterrows():
        # 已经有值的行跳过 并计数已经处理了多少个
        if pd.notna(row["female_percents"]) and pd.notna(row["mean_female_percent"]):
            processed += 1
            continue
    
        
        # if row["inventors_firstname"] is nan, skip
        if pd.isna(row["inventors_firstname"]):
            continue
        
        names = eval(row["inventors_firstname"])

        percents = []
        for name in names:
            name = name.split(" ")[0].lower()
            p = get_query_from_facebook(name, solr_url, username, password)
            percents.append(p)

        df.at[idx, "female_percents"] = str(percents)  # 存为字符串以保证能写入 CSV
        df.at[idx, "mean_female_percent"] = sum(percents)/len(percents) if percents else None

        processed += 1
        
        # 每处理一行打印一个小标记，不换行
        print("✔", end="", flush=True)

        # 每 batch_size 保存一次
        if processed % batch_size == 0:
            df.to_csv(output_csv, index=False)
            print(f"已保存 {processed} 行结果到 {output_csv}")

    # 最终保存一次
    df.to_csv(output_csv, index=False)
    print(f"处理完成，总共保存 {processed} 行结果到 {output_csv}")


# 使用示例
if __name__ == "__main__":
    solr_url = 'https://solr-facebooknames.totosearch.org/solr/facebook_lastname/select'
    username = 'research'
    password = 'insead123456'
    
    # process input and output as args
    # e.g., python get_gender_feature.py input.csv output.csv 
    if len(sys.argv) < 3:
        print("Usage: python get_gender_feature.py <input_csv> <output_csv>")   
    input_csv = sys.argv[1]  
    output_csv = sys.argv[2]

    process_csv(input_csv, output_csv, solr_url, username, password, batch_size=1000)
