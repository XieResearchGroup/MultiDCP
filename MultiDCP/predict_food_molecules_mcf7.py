"""
Simplified inference script to predict MCF7 perturbation effects from food molecules
Uses ensemble of 3 pretrained models: 1013rand1.pt, 1013rand2.pt, 1013rand4.pt
"""

import os
import sys
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'MultiDCP/models'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'MultiDCP/utils'))

import multidcp
import data_utils
from multidcp_ae_utils import initialize_model_registry

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# File paths
FOOD_MOLECULES_FILE = "MultiDCP_data/food_molecules/table_of_compounds_inchikey_name_smiles.csv"
CELL_EXPRESSION_FILE = "MultiDCP_data/data/adjusted_ccle_tcga_ad_tpm_log2.csv"
GENE_FILE = "data/gene_vector.csv"
MODEL_PATHS = [
    "saved_models/1013rand1.pt",
    "saved_models/1013rand2.pt",
    "saved_models/1013rand4.pt"
]
OUTPUT_FILE = "MultiDCP_data/predictions/food_molecules_mcf7_predictions.csv"

# Dosages to test (all available)
DOSAGES = ["0.04 um", "0.12 um", "0.37 um", "1.11 um", "3.33 um", "10.0 um"]

def load_mcf7_expression():
    """Load MCF7 cell line expression from the adjusted CCLE/TCGA file"""
    print("Loading MCF7 expression data...")
    # Try different encodings
    try:
        df = pd.read_csv(CELL_EXPRESSION_FILE, index_col=0, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(CELL_EXPRESSION_FILE, index_col=0, encoding='latin-1')

    # Find MCF7 (might be named slightly differently)
    mcf7_rows = [idx for idx in df.index if 'MCF7' in idx.upper()]
    if not mcf7_rows:
        raise ValueError("MCF7 not found in cell line expression file")

    mcf7_expression = df.loc[mcf7_rows[0]].values.astype(np.float64)
    print(f"Found MCF7: {mcf7_rows[0]}, expression shape: {mcf7_expression.shape}")
    return torch.from_numpy(mcf7_expression).to(DEVICE).double()

def load_food_molecules():
    """Load food molecules with SMILES strings"""
    print("Loading food molecules...")
    df = pd.read_csv(FOOD_MOLECULES_FILE)
    print(f"Loaded {len(df)} food molecules")
    return df

def create_dosage_encoding(dosage_str, all_dosages):
    """Create one-hot encoding for dosage"""
    encoding = torch.zeros(len(all_dosages)).to(DEVICE).double()
    if dosage_str in all_dosages:
        encoding[all_dosages.index(dosage_str)] = 1.0
    return encoding

def load_model(model_path):
    """Load a pretrained MultiDCP model"""
    print(f"Loading model from {model_path}...")

    # Initialize model parameters
    model_param_registry = initialize_model_registry()
    model_param_registry.update({
        'num_gene': 978,
        'pert_idose_input_dim': len(DOSAGES),
        'dropout': 0.3,
        'linear_encoder_flag': False
    })

    # Create model
    model = multidcp.MultiDCP_AE(device=DEVICE, model_param_registry=model_param_registry)
    model.init_weights(pretrained=None)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model = model.double()
    model.eval()

    return model

def predict_single_compound(models, gene_embeddings, mcf7_expression, smiles, dosage, compound_name):
    """Predict perturbation effect for a single compound-dosage combination using ensemble"""
    try:
        # Convert SMILES to molecular features
        drug_features = data_utils.convert_smile_to_feature([smiles], DEVICE)
        mask = data_utils.create_mask_feature(drug_features, DEVICE)

        # Prepare inputs
        cell_feature = mcf7_expression.unsqueeze(0)  # Add batch dimension
        pert_idose = create_dosage_encoding(dosage, DOSAGES).unsqueeze(0)  # Add batch dimension

        # Get predictions from all models
        predictions = []
        with torch.no_grad():
            for model in models:
                predict, _ = model(
                    input_cell_gex=cell_feature,
                    input_drug=drug_features,
                    input_gene=gene_embeddings,
                    mask=mask,
                    input_pert_idose=pert_idose,
                    job_id='perturbed'
                )
                predictions.append(predict.cpu().numpy())

        # Average predictions across models
        avg_prediction = np.mean(predictions, axis=0).flatten()

        return avg_prediction, True

    except Exception as e:
        print(f"Error predicting {compound_name} at {dosage}: {e}")
        return None, False

def main():
    print("="*80)
    print("Food Molecules MCF7 Perturbation Prediction")
    print("="*80)

    # Load data
    mcf7_expression = load_mcf7_expression()
    food_molecules_df = load_food_molecules()

    # Load gene embeddings
    print("Loading gene embeddings...")
    gene_embeddings = data_utils.read_gene(GENE_FILE, DEVICE)
    print(f"Gene embeddings shape: {gene_embeddings.shape}")

    # Load all models
    models = []
    for model_path in MODEL_PATHS:
        models.append(load_model(model_path))
    print(f"Loaded {len(models)} models for ensemble prediction")

    # Prepare results storage
    results = []

    # Predict for each compound and dosage
    print("\nStarting predictions...")
    total_predictions = len(food_molecules_df) * len(DOSAGES)

    with tqdm(total=total_predictions) as pbar:
        for idx, row in food_molecules_df.iterrows():
            inchikey = row['inchikey']
            compound_name = row['common_name']
            smiles = row['isomeric_smiles']

            for dosage in DOSAGES:
                prediction, success = predict_single_compound(
                    models, gene_embeddings, mcf7_expression,
                    smiles, dosage, compound_name
                )

                if success:
                    # Store result with metadata
                    result = {
                        'inchikey': inchikey,
                        'compound_name': compound_name,
                        'smiles': smiles,
                        'cell_line': 'MCF7',
                        'dosage': dosage,
                    }
                    # Add gene expression predictions (978 genes)
                    for gene_idx, gene_value in enumerate(prediction):
                        result[f'gene_{gene_idx}'] = gene_value

                    results.append(result)

                pbar.update(1)

    # Save results
    print(f"\nSaving results to {OUTPUT_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nPrediction complete!")
    print(f"Total predictions: {len(results_df)}")
    print(f"Results saved to: {OUTPUT_FILE}")
    print("="*80)

if __name__ == '__main__':
    main()
