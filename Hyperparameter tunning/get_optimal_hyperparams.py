def get_optimal_hyperparameters(csv_file_path):
    """
    This function takes the path to a CSV file containing model performance data and returns
    the optimal hyperparameters based on the criteria of preferring higher batch size and higher
    learning rate, provided that the mAP is not significantly different.

    Args:
    csv_file_path (str): The file path to the CSV file containing the model data.

    Returns:
    dict: A dictionary containing the optimal learning rate (LR), batch size, and mAP.
    """

    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Clean the Batch_Size column to be numeric
    df['Batch_Size'] = df['Batch_Size'].str.replace('BS_', '').astype(int)

    # Define the threshold as within 1% of the highest mAP
    max_map = df['mAP'].max()
    map_threshold = max_map * 0.99

    # Filter rows where mAP is within the 1% threshold of the max mAP
    df_filtered = df[df['mAP'] >= map_threshold]

    # Sort by Batch_Size and LR in descending order and take the first row
    best_row = df_filtered.sort_values(by=['Batch_Size', 'LR'], ascending=[False, False]).iloc[0]

    # Return the best hyperparameters as a dictionary
    return best_row[['LR', 'Batch_Size', 'mAP']].to_dict()

