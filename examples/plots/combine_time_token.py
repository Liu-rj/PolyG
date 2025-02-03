import pandas as pd

# dataset = "academia"
# dataset = "literature"
dataset = "ecommerce"

# Load the CSV file using tab as the delimiter
file_path = f'data/{dataset}.csv'  # Replace with your file path
data = pd.read_csv(file_path, delimiter='\t', header=None)

# Identify the start of latency and token count tables
latency_start = 0
tokens_start = data[data[0] == "token counts"].index[0] + 1

# Extract the headers and rows for both sections
latency_headers = data.iloc[latency_start, :].values
latency_data = data.iloc[latency_start + 1 : tokens_start - 1, :]

token_headers = data.iloc[tokens_start - 1, :].values
token_data = data.iloc[tokens_start:, :]

# Align the headers (replace "token counts" with "e2e time (s)" to match)
token_headers[0] = latency_headers[0]

# Combine both tables into one (time/token format)
combined = latency_data.copy()
for row_idx in range(latency_data.shape[0]):
    for col_idx in range(1, latency_data.shape[1]):  # Skip the first column (category names)
        latency_value = latency_data.iloc[row_idx, col_idx]
        token_value = token_data.iloc[row_idx, col_idx]
        combined.iloc[row_idx, col_idx] = f"{latency_value}/{token_value}"

combined.replace("nan/nan", "N/A", inplace=True)

# Update headers and reset the index
combined.columns = latency_headers
combined.reset_index(drop=True, inplace=True)

# Save the combined table to a new CSV file
output_path = f'data/combined_{dataset}.csv'  # Replace with your desired output path
combined.to_csv(output_path, index=False)

print(f"Combined table saved to {output_path}")
