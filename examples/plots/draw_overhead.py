import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# Load the datasets
file_ecommerce = "data/ecommerce.csv"  # Replace with your file paths
file_academia = "data/academia.csv"
file_literature = "data/literature.csv"

# labels = ["Ent-Abs", "Ent-Con", "Rel-Abs", "Rel-Con"]
labels = ["<s,*,*>", "<s,p,*>", "<s,*,o>", "<s,p,o>"]

data_ecommerce = pd.read_csv(file_ecommerce, delimiter="\t", header=None)
data_academia = pd.read_csv(file_academia, delimiter="\t", header=None)
data_literature = pd.read_csv(file_literature, delimiter="\t", header=None)


# Function to process datasets into time and token components
def process_dataset_adjusted(data):
    # Split into latency and token count sections
    mid_point = len(data) // 2
    latency_headers = data.iloc[0, :].values
    latency_data = data.iloc[1:mid_point, :]
    token_data = data.iloc[mid_point + 1 :, :]

    # Extract selected and overhead columns for time and token
    selected_col_idx = list(latency_headers).index("selected")
    overhead_col_idx = list(latency_headers).index("overhead")
    row_identifiers = latency_data.iloc[:, 0]

    selected_time = latency_data.iloc[:, selected_col_idx].astype(float)
    overhead_time = latency_data.iloc[:, overhead_col_idx].astype(float)
    selected_token = (
        token_data.iloc[:, selected_col_idx].replace(",", "", regex=True).astype(float)
    )
    overhead_token = (
        token_data.iloc[:, overhead_col_idx].replace(",", "", regex=True).astype(float)
    )

    return row_identifiers, selected_time, overhead_time, selected_token, overhead_token


# Process datasets
datasets = ["Academia", "Literature", "E-commerce"]
row_ids1, selected_time1, overhead_time1, selected_token1, overhead_token1 = (
    process_dataset_adjusted(data_academia)
)
row_ids2, selected_time2, overhead_time2, selected_token2, overhead_token2 = (
    process_dataset_adjusted(data_literature)
)
row_ids3, selected_time3, overhead_time3, selected_token3, overhead_token3 = (
    process_dataset_adjusted(data_ecommerce)
)

# Combine datasets with dataset labels
combined_data = pd.DataFrame(
    {
        "Dataset": ([datasets[0]] * len(row_ids1))
        + ([datasets[1]] * len(row_ids2))
        + ([datasets[2]] * len(row_ids3)),
        "Row Identifier": pd.concat([row_ids1, row_ids2, row_ids3]).reset_index(
            drop=True
        ),
        "Selected Time": pd.concat(
            [selected_time1, selected_time2, selected_time3]
        ).reset_index(drop=True),
        "Overhead Time": pd.concat(
            [overhead_time1, overhead_time2, overhead_time3]
        ).reset_index(drop=True),
        "Selected Token": pd.concat(
            [selected_token1, selected_token2, selected_token3]
        ).reset_index(drop=True),
        "Overhead Token": pd.concat(
            [overhead_token1, overhead_token2, overhead_token3]
        ).reset_index(drop=True),
    }
)

# Prepare data for grouped bar plots
unique_datasets = combined_data["Dataset"].unique()
x_positions = []
x_labels = labels * 3
current_position = 0

# Assign positions for bars
for dataset in unique_datasets:
    subset = combined_data[combined_data["Dataset"] == dataset]
    x_positions.extend(list(range(current_position, current_position + len(subset))))
    # x_labels.extend(subset["Row Identifier"])
    current_position += len(subset) + 1  # Add gap between datasets

FONT_SIZE = 24

# Plot grouped time chart with dataset labels
plt.figure(figsize=(14, 6))
selected_time_values = combined_data["Selected Time"]
overhead_time_values = combined_data["Overhead Time"]
plt.bar(x_positions, selected_time_values, label="Execution Time", color="dodgerblue")
plt.bar(
    x_positions,
    overhead_time_values,
    bottom=selected_time_values,
    label="Overhead Time",
    color="orange",
)

# Add centered dataset labels
for dataset in unique_datasets:
    subset = combined_data[combined_data["Dataset"] == dataset]
    start_idx = x_positions[subset.index[0]]  # Start position of the dataset group
    end_idx = x_positions[subset.index[-1]]  # End position of the dataset group
    mid_position = (start_idx + end_idx) / 2  # Calculate the midpoint
    plt.text(
        mid_position,
        -max(selected_time_values) * 0.4,
        dataset,
        ha="center",
        va="top",
        fontsize=FONT_SIZE,
        color="black",
    )

plt.xticks(x_positions, x_labels, fontsize=FONT_SIZE, rotation=45, ha="right")
plt.yticks(fontsize=FONT_SIZE)
# plt.xlabel("Row Identifiers (Grouped by Dataset)")
plt.ylabel("End-to-end Latency (s)", fontsize=FONT_SIZE)
# plt.title("Grouped Stacked Bar Chart for Time (Selected and Overhead)", fontsize=FONT_SIZE)
plt.legend(fontsize=FONT_SIZE)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("figures/grouped_time_overhead.pdf", bbox_inches="tight")

# Plot grouped token chart with dataset labels
plt.figure(figsize=(14, 6))
selected_token_values = combined_data["Selected Token"]
overhead_token_values = combined_data["Overhead Token"]
plt.bar(x_positions, selected_token_values, label="Execution Token", color="dodgerblue")
plt.bar(
    x_positions,
    overhead_token_values,
    bottom=selected_token_values,
    label="Overhead Token",
    color="orange",
)

# Add centered dataset labels
for dataset in unique_datasets:
    subset = combined_data[combined_data["Dataset"] == dataset]
    start_idx = x_positions[subset.index[0]]  # Start position of the dataset group
    end_idx = x_positions[subset.index[-1]]  # End position of the dataset group
    mid_position = (start_idx + end_idx) / 2  # Calculate the midpoint
    plt.text(
        mid_position,
        -max(selected_token_values) * 0.4,
        dataset,
        ha="center",
        va="top",
        fontsize=FONT_SIZE,
        color="black",
    )

plt.xticks(x_positions, x_labels, fontsize=FONT_SIZE, rotation=45, ha="right")
plt.yticks(fontsize=FONT_SIZE)
# plt.xlabel("Row Identifiers (Grouped by Dataset)")
plt.ylabel("Token Usage", fontsize=FONT_SIZE)
# plt.title("Grouped Stacked Bar Chart for Token (Selected and Overhead)")
plt.legend(fontsize=FONT_SIZE)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("figures/grouped_token_overhead.pdf", bbox_inches="tight")
