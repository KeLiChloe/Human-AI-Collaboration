import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

import pandas as pd

# List of file paths
file_paths = [
    "data/samples/race-pos/agency_scored_ml_sample.csv",
    "data/samples/gender-pos/agency_scored_ml_sample.csv",
    "data/samples/econ-pos/agency_scored_ml_sample.csv"
]

# Read and concatenate all files
df_list = [pd.read_csv(fp) for fp in file_paths]
df_combined = pd.concat(df_list, ignore_index=True)


# Clean and scale
df_cleaned = df_combined[df_combined['agency_score'] > 0].copy()
df_cleaned['semantic_agency'] *= 50

# Define columns and relabel
columns = ['syntactic_agency', 'semantic_agency']
df_cleaned['mentions_inequality'] = df_cleaned['AI_label'].map({
    1: 'Mentions Inequality',
    0: 'Does Not Mention Inequality'
})

# Outlier removal function
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]

# Remove outliers
filtered_df = df_cleaned.copy()
for col in columns:
    filtered_df = remove_outliers_iqr(filtered_df, col)

# Melt for combined plotting
melted_df = filtered_df.melt(id_vars='mentions_inequality', value_vars=columns,
                             var_name='Metric', value_name='Value')

# Mean values per group and metric
mean_df = melted_df.groupby(['Metric', 'mentions_inequality'])['Value'].mean().reset_index()

# T-tests for each metric
group_1 = filtered_df[filtered_df['AI_label'] == 1]
group_0 = filtered_df[filtered_df['AI_label'] == 0]
p_values = {
    col: ttest_ind(group_1[col], group_0[col], equal_var=False).pvalue
    for col in columns
}

# Get significance stars
def get_significance_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'

# Plot both metrics in one figure
sns.set(style="whitegrid")
plt.figure(figsize=(10, 7))
ax = sns.boxplot(data=melted_df, x='Metric', y='Value', hue='mentions_inequality', palette="Set2")

# Add red dotted lines for mean
for i, metric in enumerate(columns):
    group_means = mean_df[mean_df['Metric'] == metric].reset_index(drop=True)
    for j, row in enumerate(group_means.itertuples()):
        x_pos = i + (-0.2 if j == 0 else 0.2)  # offset left/right
        ax.plot([x_pos - 0.1, x_pos + 0.1], [row.Value, row.Value],
                color='red', linestyle='--', linewidth=1)
        ax.text(x_pos, row.Value, f"{row.Value:.2f}", color='red',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# Title with combined p-values
p_synt = p_values['syntactic_agency']
p_sem = p_values['semantic_agency']
if len(file_paths) == 1:
    file_path = file_paths[0]
    theme = 'Gender' if 'gender' in file_path else ('Econ' if 'econ' in file_path else 'Racial')
else:
    theme = 'Overall'

title = (
    f"Comparison of Agency by Mentions of $\\mathbf{{{theme}}}$ Inequality\n"
    f"Syntactic: p = {p_synt:.1e} {get_significance_stars(p_synt)}, "
    f"Semantic: p = {p_sem:.1e} {get_significance_stars(p_sem)}"
)
plt.title(title, fontsize=14)

# Custom legend including red line
handles, labels = ax.get_legend_handles_labels()
handles.append(plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1))
labels.append("Red dotted line: mean value")

plt.legend(handles, labels, title="Group", loc="upper right")
plt.xlabel("")  # No x-axis label
plt.ylabel("Value", fontsize=12)
plt.tight_layout(pad=2.0)
plt.show()
