from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Define prediction and ground truth vectors
Claude = {
   "social_science": +1,
   "female_score_avg": +1,
   "authors_race_diversity_score": +1,
   "country_race_diversity_score": 0,
   "black": +1,
   "natural_science": 0,
   "first_author_female_score": 0,
   "white": 0,
   "female_score_max": 0,
   "female_score_min": 0,
   "asian": 0,
   "num_authors": 0,
   "native_hawaiian": 0,
   "engineering_and_technology": -1,
   "hispanic": 0
}

GPT = {
    "social_science": +1,
    "female_score_avg": +1,
    "authors_race_diversity_score": +1,
    "country_race_diversity_score": 0,
    "black": +1,
    "natural_science": 0,
    "first_author_female_score": +1,
    "white": 0,
    "female_score_max": 0,
    "female_score_min": 0,
    "asian": 0,
    "num_authors": 0,
    "native_hawaiian": 0,
    "engineering_and_technology": 0,
    "hispanic": 0
}

Gemini = {
    "social_science": 1,
    "female_score_avg": 1,
    "authors_race_diversity_score": 1,
    "country_race_diversity_score": 0,
    "black": 1,
    "natural_science": 0,
    "first_author_female_score": 0, # Though related to my #4, I prioritized female_score_avg for the distinct top 5
    "white": 0,
    "female_score_max": 0,
    "female_score_min": 0,
    "asian": 0,
    "num_authors": 0,
    "native_hawaiian": 0,
    "engineering_and_technology": 0,
    "hispanic": 1
}

DeepSeek = {
    "social_science": +1,
    "female_score_avg": 0,
    "authors_race_diversity_score": +1,
    "country_race_diversity_score": 0,
    "black": +1,
    "natural_science": 0,
    "first_author_female_score": +1,
    "white": 0,
    "female_score_max": 0,
    "female_score_min": 0,
    "asian": 0,
    "num_authors": 0,
    "native_hawaiian": 0,
    "engineering_and_technology": -1,
    "hispanic": 0
}

ground_truth = {
    "social_science": 1,
    "female_score_avg": 1,
    "country_race_diversity_score": 1,
    "female_score_max": 1,
    "female_score_min": -1,
    "authors_race_diversity_score": 0,
    "black": 0,
    "natural_science": 0,
    "first_author_female_score": 0,
    "white": 0,
    "asian": 0,
    "num_authors": 0,
    "native_hawaiian": 0,
    "engineering_and_technology": 0,
    "hispanic": 0
}


# Ensure consistent feature order
features = list(ground_truth.keys())

def sum_squared_distance(vec1, vec2):
    return np.sum((vec1 - vec2) ** 2)

# Convert to numpy arrays
def to_vector(model_dict):
    return np.array([model_dict.get(key, 0) for key in features]).reshape(1, -1)

vec_claude = to_vector(Claude)
vec_gpt = to_vector(GPT)
vec_gemini = to_vector(Gemini)
vec_deepseek = to_vector(DeepSeek)
vec_truth = to_vector(ground_truth)

# Compute cosine similarities
sim_claude = cosine_similarity(vec_claude, vec_truth)[0][0]
sim_gpt = cosine_similarity(vec_gpt, vec_truth)[0][0]
sim_gemini = cosine_similarity(vec_gemini, vec_truth)[0][0]
sim_deepseek = cosine_similarity(vec_deepseek, vec_truth)[0][0]

#  compute sum squared distances
ssd_claude = sum_squared_distance(vec_claude, vec_truth)
ssd_gpt = sum_squared_distance(vec_gpt, vec_truth)
ssd_gemini = sum_squared_distance(vec_gemini, vec_truth)
ssd_deepseek = sum_squared_distance(vec_deepseek, vec_truth)

print(f"Claude similarity: {sim_claude:.2f}")
print(f"GPT similarity: {sim_gpt:.2f}")
print(f"Gemini similarity: {sim_gemini:.2f}")
print(f"DeepSeek similarity: {sim_deepseek:.2f}")

print(f"Claude sum squared distance: {ssd_claude:.2f}")
print(f"GPT sum squared distance: {ssd_gpt:.2f}")
print(f"Gemini sum squared distance: {ssd_gemini:.2f}")
print(f"DeepSeek sum squared distance: {ssd_deepseek:.2f}")