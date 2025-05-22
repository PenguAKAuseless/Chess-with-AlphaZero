import argparse
import os
import logging
import anndata as ad
import torch
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast
from sklearn.cluster import KMeans
import numpy as np
from scGPT_model import scGPT
import json
import time
import matplotlib.pyplot as plt
import umap

# Custom Dataset for AnnData (copied from trainer.py to make clustering.py self-contained)
class AnnDataDataset(Dataset):
    def __init__(self, adata, gene_ids, max_seq_len=512):
        self.adata = adata
        self.max_seq_len = max_seq_len
        self.gene_ids = gene_ids
        
        gene_to_idx = {gene: idx for idx, gene in enumerate(self.gene_ids)}
        common_genes = [g for g in adata.var.index if g in gene_to_idx]
        
        self.expression = adata[:, common_genes].X
        self.gene_indices = np.array([gene_to_idx[g] for g in common_genes])
        
        self.batch_ids = adata.obs['donor_id'].cat.codes.values if 'donor_id' in adata.obs else np.zeros(len(adata), dtype=int)
        self.condition_tokens = adata.obs['Dataset'].cat.codes.values if 'Dataset' in adata.obs else np.zeros(len(adata), dtype=int)

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        expr = self.expression[idx].toarray()[0] if hasattr(self.expression, 'toarray') else self.expression[idx]
        gene_indices = self.gene_indices[:self.max_seq_len]
        expr = expr[:self.max_seq_len]
        batch_id = self.batch_ids[idx]
        cond = self.condition_tokens[idx]
        
        return {
            'gene_tokens': torch.tensor(gene_indices, dtype=torch.long),
            'expression_values': torch.tensor(expr, dtype=torch.float),
            'condition_tokens': torch.tensor(cond, dtype=torch.long),
            'batch_ids': torch.tensor(batch_id, dtype=torch.long),
            'modality_tokens': torch.zeros(1, dtype=torch.long)
        }

# Clustering Function
def cluster_data(model, h5ad_files, gene_ids, device, output_dir, max_seq_len):
    model.eval()
    logging.info("Starting clustering across all datasets")
    start_time = time.time()
    embeddings = []
    total_cells = 0
    cell_metadata = []
    
    with torch.no_grad():
        for h5ad_file in h5ad_files:
            logging.info(f"Loading dataset for clustering: {h5ad_file}")
            adata = ad.read_h5ad(h5ad_file)
            total_cells += adata.shape[0]
            
            file_metadata = {
                'file': os.path.basename(h5ad_file),
                'cell_indices': list(range(total_cells - adata.shape[0], total_cells))
            }
            
            for col in ['donor_id', 'Dataset', 'cell_type']:
                if col in adata.obs:
                    file_metadata[col] = adata.obs[col].astype(str).tolist()
            
            cell_metadata.append(file_metadata)
            
            dataset = AnnDataDataset(adata, gene_ids, max_seq_len)
            loader = DataLoader(dataset, batch_size=16, shuffle=False)
            for batch in loader:
                with autocast():
                    gene_tokens = batch['gene_tokens'].to(device)
                    expression_values = batch['expression_values'].to(device)
                    condition_tokens = batch['condition_tokens'].to(device)
                    batch_ids = batch['batch_ids'].to(device)
                    modality_tokens = batch['modality_tokens'].to(device)
                    
                    model_output = model(gene_tokens, expression_values, condition_tokens, batch_ids, modality_tokens)
                    logits = model_output[0]
                    embeddings.append(logits.cpu().numpy())
                
            logging.info(f"Processed {h5ad_file} for clustering: {adata.shape[0]} cells, {adata.shape[1]} genes")
            del adata
            torch.cuda.empty_cache()
    
    embeddings = np.concatenate(embeddings, axis=0)
    logging.info(f"Total embeddings shape: {embeddings.shape}")

    kmeans = KMeans(n_clusters=50, random_state=42).fit(embeddings)
    clusters = kmeans.labels_
    np.save(os.path.join(output_dir, "clusters.npy"), clusters)
    
    with open(os.path.join(output_dir, "cell_metadata.json"), 'w') as f:
        json.dump(cell_metadata, f)
    
    logging.info("Performing UMAP dimensionality reduction")
    reducer = umap.UMAP(random_state=42)
    umap_embeddings = reducer.fit_transform(embeddings)
    np.save(os.path.join(output_dir, "umap_embeddings.npy"), umap_embeddings)
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=clusters, cmap='viridis', s=5, alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    plt.title('UMAP Projection of scGPT Embeddings')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "umap_visualization.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "umap_visualization.pdf"))
    plt.close()
    
    if len(np.unique(clusters)) > 10:
        logging.info("Creating cluster-specific visualizations")
        plt.figure(figsize=(15, 15))
        for cluster_id in range(min(16, len(np.unique(clusters)))):
            plt.subplot(4, 4, cluster_id + 1)
            mask = clusters == cluster_id
            plt.scatter(umap_embeddings[~mask, 0], umap_embeddings[~mask, 1], 
                        c='lightgrey', s=1, alpha=0.1)
            plt.scatter(umap_embeddings[mask, 0], umap_embeddings[mask, 1], 
                        c='red', s=3, alpha=0.7)
            plt.title(f'Cluster {cluster_id}')
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "cluster_highlights.png"), dpi=300)
        plt.close()
    
    total_time = time.time() - start_time
    logging.info(f"Clustering and visualization completed. Total cells: {total_cells}, "
                 f"Results saved to {output_dir}, Time: {total_time:.2f}s")
    return clusters, umap_embeddings

def main():
    parser = argparse.ArgumentParser(description="Cluster scGPT embeddings")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with .h5ad files')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model file')
    parser.add_argument('--mapping_dir', type=str, required=True, help='Directory with gene mappings')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save clustering results')
    parser.add_argument('--log_dir', type=str, required=True, help='Directory to save logs')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--max_seq_len', type=int, default=512, help='Maximum sequence length')

    args = parser.parse_args()

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_filename = f'clustering_{timestamp}.log'
    os.makedirs(args.log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.log_dir, log_filename),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("Starting scGPT clustering pipeline")

    h5ad_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith('.h5ad')]
    if not h5ad_files:
        logging.error("No .h5ad files found in input_dir")
        return

    gene_file = os.path.join(args.mapping_dir, "gene_ids.txt")
    if not os.path.exists(gene_file):
        logging.error(f"Gene mappings not found at {gene_file}")
        return
    with open(gene_file, 'r') as f:
        gene_ids = [line.strip() for line in f]

    checkpoint = torch.load(args.model_path)
    model_metadata = checkpoint['metadata']
    vocab_size = model_metadata['vocab_size']
    condition_size = model_metadata['condition_size']
    batch_size = model_metadata['batch_size']
    modality_size = model_metadata['modality_size']
    d_model = model_metadata['d_model']
    n_heads = model_metadata['n_heads']
    n_layers = model_metadata['n_layers']
    d_ff = model_metadata['d_ff']
    max_seq_len = model_metadata['max_seq_len']
    dropout = model_metadata['dropout']

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')
    logging.info(f"Using device: {device}")

    model = scGPT(vocab_size, condition_size, batch_size, modality_size,
                  d_model, n_heads, n_layers, d_ff, max_seq_len, dropout).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Loaded model from {args.model_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    clusters, umap_embeddings = cluster_data(model, h5ad_files, gene_ids, device, args.output_dir, args.max_seq_len)
    
    logging.info("scGPT clustering pipeline completed")

if __name__ == "__main__":
    main()