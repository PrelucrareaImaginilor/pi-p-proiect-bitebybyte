import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm  # Import tqdm for progress bar

# Paths
metadata_path = 'C:/Users/silvi/Downloads/widsdatathon2025-university/metadata/training_metadata.csv'
matrix_folder = 'C:/Users/silvi/Downloads/widsdatathon2025-university/train_tsv/train_tsv'

metadata_test_path = 'C:/Users/silvi/Downloads/widsdatathon2025-university/metadata/test_metadata.csv'
matrix_test ='C:/Users/silvi/Downloads/widsdatathon2025-university/test_tsv/test_tsv'

# Load metadata
print("Loading metadata...")
metadata_df = pd.read_csv(metadata_test_path)
print(f"Metadata loaded. Total records: {len(metadata_df)}")

# Specify categorical and numerical columns in the metadata, excluding 'ethnicity'
categorical_features = ['sex', 'study_site', 'race', 'handedness', 'parent_1_education', 'parent_2_education']
numerical_features = ['bmi', 'p_factor_fs', 'internalizing_fs', 'externalizing_fs', 'attention_fs']
target = 'age'  # Define target column for potential later use

# Encoding specific categorical columns to numerical values (excluding 'ethnicity')
encoding_maps = {
    'sex': {'Male': 0, 'Female': 1}, 
    'race': {'White': 0, 'Black': 1, 'Asian': 2, 'Other': 3}  
}

print("Encoding categorical columns (sex, race)...")
# Apply the encoding maps to the relevant columns
for col, mapping in encoding_maps.items():
    metadata_df[col] = metadata_df[col].map(mapping)
print("Encoding complete.")

# Function to find the file corresponding to a specific patient ID
def find_matrix_file(patient_id, folder_path):
    for filename in os.listdir(folder_path):
        if f"sub-{patient_id}" in filename:
            return os.path.join(folder_path, filename)
    return None

# Function to load and process each patient's connectivity matrix
def process_connectivity_matrix(file_path):
    matrix = np.loadtxt(file_path, delimiter='\t')

    # Extract the upper triangular portion (excluding the diagonal)
    upper_triangular_values = matrix[np.triu_indices_from(matrix, k=1)]

    return upper_triangular_values

# List to store processed data
processed_data = []

# Loop over each patient in the metadata with a progress bar
print("Processing each patient and their connectivity matrix...")
for _, row in tqdm(metadata_df.iterrows(), total=metadata_df.shape[0], desc="Processing patients"):
    participant_id = row['participant_id']

    # Find the file containing the patient's connectivity matrix
    file_path = find_matrix_file(participant_id, matrix_test)

    if file_path is None:
        print(f"File not found for patient {participant_id}")
        continue

    # Process the connectivity matrix for this patient
    connectivity_vector = process_connectivity_matrix(file_path)

    # Combine metadata and connectivity vector for this patient
    combined_row = row.to_dict()  # Convert metadata row to dictionary

    # Add each connectivity matrix element as a separate column
    for i, value in enumerate(connectivity_vector):
        combined_row[f'conn_{i}'] = value

    processed_data.append(combined_row)

print("Finished processing all patients.")

# Create a DataFrame with the processed data
print("Creating DataFrame from processed data...")
final_df = pd.DataFrame(processed_data)
print("DataFrame created.")

# Apply one-hot encoding to remaining categorical features
print("Applying one-hot encoding to categorical features (study_site, handedness, parent_1_education, parent_2_education)...")
final_df = pd.get_dummies(final_df, columns=[col for col in categorical_features if col not in encoding_maps], drop_first=True)
print("One-hot encoding complete.")

# Standardize numerical features
print("Standardizing numerical features...")
scaler = StandardScaler()
final_df[numerical_features] = scaler.fit_transform(final_df[numerical_features])
print("Standardization complete.")

# Preserve participant_id and drop all remaining non-numeric columns
print("Removing non-numeric columns...")
participant_id = final_df['participant_id'] 
final_df = final_df.select_dtypes(include=[np.number]) 
final_df['participant_id'] = participant_id  # Add participant_id back
print("Non-numeric columns removed. Final data is numerical only, including participant_id.")

# Save the merged DataFrame to a CSV file for future use
print("Saving the processed data to 'merged_data2.csv'...")
final_df.to_csv('merged_test.csv', index=False)
print("Data processing complete. Merged data saved to 'merged_data2.csv'")
