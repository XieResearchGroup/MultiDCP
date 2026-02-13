#!/usr/bin/env python3
"""
Script to add gene names to MultiDCP prediction files.

This script replaces generic column names (gene_0, gene_1, ..., gene_977)
with actual gene names (DDR1, PAX8, RPS5, ..., NPEPL1) for better interpretability.

Usage:
    python add_gene_names_to_predictions.py
"""

import pandas as pd
import sys

def main():
    # File paths
    gene_vector_file = "data/gene_vector.csv"
    input_predictions = "MultiDCP_data/predictions/food_molecules_mcf7_predictions.csv"
    output_predictions = "MultiDCP_data/predictions/food_molecules_mcf7_predictions_with_gene_names.csv"

    print("Loading gene names from gene_vector.csv...")
    # Read gene names from gene_vector.csv (first column, no header)
    gene_df = pd.read_csv(gene_vector_file, header=None)
    gene_names = gene_df.iloc[:, 0].tolist()

    print(f"Found {len(gene_names)} gene names")
    print(f"First 5 genes: {gene_names[:5]}")
    print(f"Last 5 genes: {gene_names[-5:]}")

    print("\nLoading prediction file...")
    # Read the predictions file
    predictions_df = pd.read_csv(input_predictions)

    print(f"Predictions file shape: {predictions_df.shape}")
    print(f"Columns: {list(predictions_df.columns[:10])}...")

    # Count gene columns in predictions
    gene_columns = [col for col in predictions_df.columns if col.startswith('gene_')]
    print(f"Number of gene columns in predictions: {len(gene_columns)}")

    # Check if counts match
    if len(gene_names) != len(gene_columns):
        print(f"\nWARNING: Mismatch detected!")
        print(f"  Gene names in gene_vector.csv: {len(gene_names)}")
        print(f"  Gene columns in predictions: {len(gene_columns)}")

        # If we have fewer gene names, we need to handle this
        if len(gene_names) < len(gene_columns):
            print(f"\nPadding gene names list with placeholder for missing gene...")
            # Add a placeholder for the missing gene
            gene_names.append(f"GENE_{len(gene_names)}")

    # Create mapping from old column names to new gene names
    print("\nCreating column name mapping...")
    column_mapping = {}
    for i, gene_col in enumerate(gene_columns):
        if i < len(gene_names):
            column_mapping[gene_col] = gene_names[i]
        else:
            # Should not happen if we padded correctly
            column_mapping[gene_col] = f"UNKNOWN_GENE_{i}"

    # Rename columns
    print("Renaming gene columns...")
    predictions_df_renamed = predictions_df.rename(columns=column_mapping)

    # Verify the rename worked
    print(f"\nNew column names (first 10): {list(predictions_df_renamed.columns[:10])}")
    print(f"Last 5 columns: {list(predictions_df_renamed.columns[-5:])}")

    # Save to new file
    print(f"\nSaving to {output_predictions}...")
    predictions_df_renamed.to_csv(output_predictions, index=False)

    print(f"\nSuccess! Created {output_predictions}")
    print(f"  Rows: {len(predictions_df_renamed)}")
    print(f"  Columns: {len(predictions_df_renamed.columns)}")
    print(f"  First few gene columns: {[col for col in predictions_df_renamed.columns if col in gene_names[:5]]}")
    print(f"  Last few gene columns: {[col for col in predictions_df_renamed.columns if col in gene_names[-5:]]}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
