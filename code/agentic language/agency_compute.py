from tqdm import tqdm
import os
import pandas as pd
import spacy
from empath import Empath

# Load models
nlp = spacy.load("en_core_web_sm")
lexicon = Empath()

# Load your CSV
input_path = "/Users/like/Desktop/Research/Human-AI/data/samples/econ-pos/updated_data_with_label_refinement.csv"  # replace with your actual file path
df = pd.read_csv(input_path)

# Register tqdm
tqdm.pandas()

# Define syntactic agency function
def syntactic_agency(text):
    doc = nlp(str(text))
    sentences = list(doc.sents)
    if not sentences:
        return 0.0
    passive_sentences = sum(1 for sent in sentences if any(tok.dep_ == "auxpass" for tok in sent))
    return 1 - (passive_sentences / len(sentences))

# Define semantic agency function
def semantic_agency(text):
    categories = ["power", "achievement", "leader", "dominant_personality", "gain", "strength"]
    scores = lexicon.analyze(str(text), categories=categories, normalize=True)
    # deal with the case scores is None
    if scores is None:
        avg_score = -100
    else:
        avg_score = sum(scores.get(c, 0) for c in categories) # / len(categories)
    return avg_score

# Combine into final score
def compute_agency_components(text, w1=1, w2=5):
    sa = syntactic_agency(text)
    sema = semantic_agency(text)
    score = w1 * sa + w2 * sema
    return pd.Series({
        "syntactic_agency": sa,
        "semantic_agency": sema,
        "agency_score": score
    })


# Apply all in one progress bar
df[["syntactic_agency", "semantic_agency", "agency_score"]] = df["paper_abstract"].progress_apply(compute_agency_components)

# Create output path in the same directory
input_dir = os.path.dirname(input_path)
output_path = os.path.join(input_dir, "agency_scored_ml_sample.csv")

# Save result
df.to_csv(output_path, index=False)
