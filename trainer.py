import argparse
import os
import logging
import h5py
import anndata as ad
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
import numpy as np
from scGPT_model import scGPT, StableDynamicMaskedLoss
import json
import time
from datetime import datetime

# Set PyTorch memory management to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Custom Dataset for AnnData
class AnnDataDataset(Dataset):
    def __init__(self, adata, gene_ids, max_seq_len=512):
        self.adata = adata
        self.max_seq_len = max_seq_len
        self.gene_ids = gene_ids
        
        # Use provided gene_ids mapping
        gene_to_idx = {gene: idx for idx, gene in enumerate(self.gene_ids)}
        common_genes = [g for g in adata.var.index if g in gene_to_idx]
        
        # Keep expression data as sparse matrix if possible
        self.expression = adata[:, common_genes].X
        self.gene_indices = np.array([gene_to_idx[g] for g in common_genes])
        
        # Handle donor_id and Dataset, ensuring they exist
        self.batch_ids = adata.obs['donor_id'].cat.codes.values if 'donor_id' in adata.obs else np.zeros(len(adata), dtype=int)
        self.condition_tokens = adata.obs['Dataset'].cat.codes.values if 'Dataset' in adata.obs else np.zeros(len(adata), dtype=int)

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        # Convert only the required slice to dense format
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

# Function to split AnnData into smaller chunks
def split_anndata(adata, chunk_size):
    """Split AnnData object into smaller chunks."""
    n_cells = adata.shape[0]
    chunks = []
    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        chunk = adata[start:end].copy()
        chunks.append(chunk)
    return chunks

# Training Function
def train_model(model, train_loader, criterion, optimizer, device, checkpoint_dir, epochs, interval, dataset_name, accum_steps=4):
    model.train()
    logging.info(f"Starting training on dataset: {dataset_name}")
    start_time = time.time()
    losses = []
    scaler = GradScaler()
    
    for epoch in range(epochs):
        total_loss = 0
        epoch_start = time.time()
        optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            if device.type == 'cuda':
                mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
                mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
            
            with autocast():
                gene_tokens = batch['gene_tokens'].to(device)
                expression_values = batch['expression_values'].to(device)
                condition_tokens = batch['condition_tokens'].to(device)
                batch_ids = batch['batch_ids'].to(device)
                modality_tokens = batch['modality_tokens'].to(device)

                model_output = model(gene_tokens, expression_values, condition_tokens, batch_ids, modality_tokens)
                loss = criterion(model_output, gene_tokens)
                loss = loss / accum_steps

            scaler.scale(loss).backward()
            total_loss += loss.item()

            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            losses.append(loss.item() * accum_steps)

        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        logging.info(f"Epoch {epoch+1}/{epochs} on {dataset_name}, Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")

        if (epoch + 1) % interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{dataset_name}_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")

    total_time = time.time() - start_time
    min_loss = min(losses) if losses else 0
    max_loss = max(losses) if losses else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    logging.info(f"Training completed on {dataset_name}. Total time: {total_time:.2f}s, "
                 f"Min Loss: {min_loss:.4f}, Max Loss: {max_loss:.4f}, Avg Loss: {avg_loss:.4f}")

# Function to load or create mappings for genes, conditions, and batches
def load_or_create_mappings(h5ad_files, mapping_dir):
    os.makedirs(mapping_dir, exist_ok=True)
    gene_file = os.path.join(mapping_dir, "gene_ids.txt")
    condition_file = os.path.join(mapping_dir, "condition_ids.txt")
    batch_file = os.path.join(mapping_dir, "batch_ids.txt")
    
    if all(os.path.exists(f) for f in [gene_file, condition_file, batch_file]):
        logging.info(f"Loading existing mappings from {mapping_dir}")
        with open(gene_file, 'r') as f:
            gene_ids = [line.strip() for line in f]
        with open(condition_file, 'r') as f:
            condition_ids = [line.strip() for line in f]
        with open(batch_file, 'r') as f:
            batch_ids = [line.strip() for line in f]
            
        logging.info(f"Loaded {len(gene_ids)} genes, {len(condition_ids)} conditions, {len(batch_ids)} batches")
        return gene_ids, condition_ids, batch_ids
    
    logging.info("Creating new mappings from datasets")
    gene_set = set()
    condition_set = set()
    batch_set = set()
    
    for h5ad_file in h5ad_files:
        logging.info(f"Extracting mappings from {h5ad_file}")
        adata = ad.read_h5ad(h5ad_file)
        
        gene_set.update(adata.var.index.values)
        if 'Dataset' in adata.obs:
            condition_set.update(adata.obs['Dataset'].cat.categories.values)
        if 'donor_id' in adata.obs:
            batch_set.update(adata.obs['donor_id'].cat.categories.values)
            
        del adata
    
    gene_ids = sorted(list(gene_set))
    condition_ids = sorted(list(condition_set))
    batch_ids = sorted(list(batch_set))
    
    with open(gene_file, 'w') as f:
        f.write('\n'.join(gene_ids))
    with open(condition_file, 'w') as f:
        f.write('\n'.join(condition_ids))
    with open(batch_file, 'w') as f:
        f.write('\n'.join(batch_ids))
        
    logging.info(f"Created and saved mappings: {len(gene_ids)} genes, {len(condition_ids)} conditions, {len(batch_ids)} batches")
    return gene_ids, condition_ids, batch_ids

def main():
    parser = argparse.ArgumentParser(description="Train scGPT model")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with .h5ad files')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--model_save_dir', type=str, required=True, help='Directory to save the final model')
    parser.add_argument('--log_dir', type=str, required=True, help='Directory to save logs')
    parser.add_argument('--mapping_dir', type=str, required=True, help='Directory to save/load mappings')
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--d_ff', type=int, default=1024, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--max_seq_len', type=int, default=512, help='Maximum sequence length')

    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'training_{timestamp}.log'
    os.makedirs(args.log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.log_dir, log_filename),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("Starting scGPT training pipeline")

    h5ad_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith('.h5ad')]
    if not h5ad_files:
        logging.error("No .h5ad files found in input_dir")
        return
    
    gene_ids, condition_ids, batch_ids = load_or_create_mappings(h5ad_files, args.mapping_dir)
    
    vocab_size = len(gene_ids) + 3
    condition_size = len(condition_ids)
    batch_size = len(batch_ids)
    modality_size = 10
    
    logging.info(f"Model parameters: vocab_size={vocab_size}, condition_size={condition_size}, "
                 f"batch_size={batch_size}, gene_count={len(gene_ids)}")

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')
    logging.info(f"Using device: {device}")
    model = scGPT(vocab_size, condition_size, batch_size, modality_size,
                  args.d_model, args.n_heads, args.n_layers, args.d_ff,
                  args.max_seq_len, args.dropout).to(device)
    criterion = StableDynamicMaskedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    for h5ad_file in h5ad_files:
        logging.info(f"Loading dataset for training: {h5ad_file}")
        start_time = time.time()
        adata = ad.read_h5ad(h5ad_file)
        logging.info(f"Loaded {h5ad_file}: {adata.shape[0]} cells, {adata.shape[1]} genes, "
                     f"load time: {time.time() - start_time:.2f}s")
        
        chunk_size = 10000
        adata_chunks = split_anndata(adata, chunk_size)
        logging.info(f"Split {h5ad_file} into {len(adata_chunks)} chunks of size {chunk_size}")
        
        for chunk_idx, adata_chunk in enumerate(adata_chunks):
            logging.info(f"Training on chunk {chunk_idx + 1}/{len(adata_chunks)}")
            dataset = AnnDataDataset(adata_chunk, gene_ids, args.max_seq_len)
            train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
            epochs = max(10, adata_chunk.shape[0] // 10000)
            interval = max(1, epochs // 10)
            train_model(model, train_loader, criterion, optimizer, device, args.checkpoint_dir, 
                        epochs, interval, f"{os.path.basename(h5ad_file)}_chunk_{chunk_idx}", accum_steps=4)
            
            del dataset, train_loader
            logging.info(f"Memory cleared for chunk {chunk_idx + 1} of {h5ad_file}")
        
        del adata, adata_chunks
        logging.info(f"Memory cleared for {h5ad_file}")
        torch.cuda.empty_cache()

    os.makedirs(args.model_save_dir, exist_ok=True)
    final_model_path = os.path.join(args.model_save_dir, 'final_model.pt')
    model_metadata = {
        'vocab_size': vocab_size,
        'condition_size': condition_size,
        'batch_size': batch_size,
        'modality_size': modality_size,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'n_layers': args.n_layers,
        'd_ff': args.d_ff,
        'max_seq_len': args.max_seq_len,
        'dropout': args.dropout,
        'timestamp': timestamp,
        'gene_count': len(gene_ids),
        'condition_count': len(condition_ids),
        'batch_count': len(batch_ids)
    }
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'metadata': model_metadata
    }, final_model_path)
    
    with open(os.path.join(args.model_save_dir, 'model_metadata.json'), 'w') as f:
        json.dump(model_metadata, f, indent=2)
        
    logging.info(f"Saved final model to {final_model_path} with metadata")
    logging.info("scGPT training pipeline completed")

if __name__ == "__main__":
    main()