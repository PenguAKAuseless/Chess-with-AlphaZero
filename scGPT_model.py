import torch
import torch.nn as nn
import math

# Custom Performer-style linear attention for efficiency
class PerformerAttention(nn.Module):
    def __init__(self, d_model, n_heads, kernel="relu"):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, f"d_model {d_model} must be divisible by n_heads {n_heads}"
        self.d_head = d_model // n_heads
        self.kernel = kernel

        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Reshape Q, K, V with consistent dimensions
        Q = self.Q(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.K(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.V(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        print(f"Q shape: {Q.shape}")  # Expected: (batch_size, n_heads, seq_len, d_head)
        print(f"K shape: {K.shape}")  # Expected: (batch_size, n_heads, seq_len, d_head)
        print(f"V shape: {V.shape}")  # Expected: (batch_size, n_heads, seq_len, d_head)   

        # Performer-style kernel approximation for fast attention
        if self.kernel == "relu":
            Q = torch.relu(Q)
            K = torch.relu(K)
            
        # Make sure K_sum is calculated with the right dimension
        K_sum = K.sum(dim=-1, keepdim=True)  # (32, 8, 512, 1)
        print(f"K_sum shape: {K_sum.shape}")  # Expected: (batch_size, n_heads, seq_len, d_head)
        
        # FIX: Ensure causal_mask has the correct shape for broadcasting
        if mask is not None:
            # Ensure mask is properly expanded for broadcasting
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # Add head dimension
                
        # Compute attention scores
        attn = torch.einsum("bhnd,bhmd->bhnm", Q, K) / (K_sum + 1e-6)
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
            
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("bhnm,bhmd->bhnd", attn, V)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.out(out), attn

# Transformer Decoder Block with Performer Attention
class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = PerformerAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, attn_scores = self.attn(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x, attn_scores

# Main scGPT Model
class scGPT(nn.Module):
    def __init__(self, vocab_size, condition_size, batch_size, modality_size, d_model=512, n_heads=8, n_layers=6, d_ff=2048, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.d_head = d_model // n_heads  # Add this line to track head dimension

        # Embeddings
        self.gene_embedding = nn.Embedding(vocab_size, d_model)
        self.condition_embedding = nn.Embedding(condition_size, d_model)
        self.batch_embedding = nn.Embedding(batch_size, d_model)
        self.modality_embedding = nn.Embedding(modality_size, d_model)

        # Fully connected layer for binned gene expression values
        self.expression_fc = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )

        # Positional encoding
        self.pos_encoding = nn.Parameter(self._init_positional_encoding(max_seq_len, d_model))

        # Transformer decoder
        self.transformer_blocks = nn.ModuleList([
            TransformerDecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)

    def _init_positional_encoding(self, max_seq_len, d_model):
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _create_dynamic_causal_mask(self, seq_len, attn_scores, device):
        batch_size, seq_len_q, seq_len_k = attn_scores.size()  # attn_scores: [batch_size, seq_len, seq_len]
        assert seq_len_q == seq_len_k == seq_len, f"Expected seq_len {seq_len}, got {seq_len_q}, {seq_len_k}"
        # Initialize mask with correct shape for attention
        mask = torch.ones(batch_size, self.n_heads, seq_len, seq_len, device=device)
        # Use attention scores to uncover tokens
        threshold = attn_scores.mean()  # Scalar
        uncover = (attn_scores > threshold).float()  # [batch_size, seq_len, seq_len]
        uncover = uncover.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
        mask = mask * uncover  # Broadcasting: [batch_size, n_heads, seq_len, seq_len]
        # Ensure causal property
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
        mask = mask * causal_mask  # [batch_size, n_heads, seq_len, seq_len]
        print(f"dyn_mask shape: {mask.shape}")  # Debug
        return mask

    def forward(self, gene_tokens, expression_values, condition_tokens, batch_ids, modality_tokens):
        # First ensure all inputs are at least 2D tensors with [batch_size, seq_len] shape
        if gene_tokens.dim() == 1:
            gene_tokens = gene_tokens.unsqueeze(0)  # Add batch dimension if missing
        
        batch_size, seq_len = gene_tokens.size()
        device = gene_tokens.device
        
        # Ensure condition_tokens has proper dimensions
        if condition_tokens.dim() == 0:  # scalar
            condition_tokens = condition_tokens.unsqueeze(0).unsqueeze(0)
        elif condition_tokens.dim() == 1:  # 1D tensor
            condition_tokens = condition_tokens.unsqueeze(0)
        
        # Ensure expression_values has proper dimensions
        if expression_values.dim() == 0:  # scalar
            expression_values = expression_values.unsqueeze(0).unsqueeze(0)
        elif expression_values.dim() == 1:  # 1D tensor
            expression_values = expression_values.unsqueeze(0)
            
        # Ensure batch_ids has proper dimensions
        if batch_ids.dim() == 0:  # scalar
            batch_ids = batch_ids.unsqueeze(0).unsqueeze(0)
        elif batch_ids.dim() == 1:  # 1D tensor
            batch_ids = batch_ids.unsqueeze(0)
            
        # Ensure modality_tokens has proper dimensions
        if modality_tokens.dim() == 0:  # scalar
            modality_tokens = modality_tokens.unsqueeze(0).unsqueeze(0)
        elif modality_tokens.dim() == 1:  # 1D tensor
            modality_tokens = modality_tokens.unsqueeze(0)

        # Now handle the sequence length dimension for all inputs
        if condition_tokens.size(1) != seq_len:
            # Option 1: Broadcast condition to match gene_tokens sequence length
            # This assumes each condition applies to all tokens in the sequence
            if condition_tokens.size(1) == 1:
                condition_tokens = condition_tokens.expand(batch_size, seq_len)
            # Option 2: Truncate or pad condition tokens to match sequence length
            else:
                # Make sure condition_tokens batch dimension matches gene_tokens
                if condition_tokens.size(0) != batch_size:
                    # Either broadcast or duplicate the condition tokens to match batch size
                    if condition_tokens.size(0) == 1:
                        condition_tokens = condition_tokens.expand(batch_size, condition_tokens.size(1))
                    else:
                        # This is the problematic case - mismatched batch sizes
                        # Create a new tensor with the correct batch size
                        new_condition_tokens = torch.zeros(
                            batch_size, condition_tokens.size(1),
                            dtype=condition_tokens.dtype,
                            device=condition_tokens.device
                        )
                        # Copy over the available data
                        min_batch = min(batch_size, condition_tokens.size(0))
                        new_condition_tokens[:min_batch] = condition_tokens[:min_batch]
                        condition_tokens = new_condition_tokens
                
                # Now handle sequence length
                condition_tokens = condition_tokens[:, :seq_len]
                # Pad if needed
                if condition_tokens.size(1) < seq_len:
                    padding = torch.zeros(
                        batch_size, seq_len - condition_tokens.size(1),
                        dtype=condition_tokens.dtype,
                        device=condition_tokens.device
                    )
                    condition_tokens = torch.cat([condition_tokens, padding], dim=1)
        
        # Apply the same fix for batch_ids
        if batch_ids.size(1) != seq_len:
            if batch_ids.size(1) == 1:
                batch_ids = batch_ids.expand(batch_size, seq_len)
            else:
                # Make sure batch_ids batch dimension matches gene_tokens
                if batch_ids.size(0) != batch_size:
                    if batch_ids.size(0) == 1:
                        batch_ids = batch_ids.expand(batch_size, batch_ids.size(1))
                    else:
                        new_batch_ids = torch.zeros(
                            batch_size, batch_ids.size(1),
                            dtype=batch_ids.dtype,
                            device=batch_ids.device
                        )
                        min_batch = min(batch_size, batch_ids.size(0))
                        new_batch_ids[:min_batch] = batch_ids[:min_batch]
                        batch_ids = new_batch_ids
                
                # Now handle sequence length
                batch_ids = batch_ids[:, :seq_len]
                if batch_ids.size(1) < seq_len:
                    padding = torch.zeros(
                        batch_size, seq_len - batch_ids.size(1),
                        dtype=batch_ids.dtype,
                        device=batch_ids.device
                    )
                    batch_ids = torch.cat([batch_ids, padding], dim=1)
                    
        # Apply the same fix for modality_tokens
        if modality_tokens.size(1) != seq_len:
            if modality_tokens.size(1) == 1:
                modality_tokens = modality_tokens.expand(batch_size, seq_len)
            else:
                # Make sure modality_tokens batch dimension matches gene_tokens
                if modality_tokens.size(0) != batch_size:
                    if modality_tokens.size(0) == 1:
                        modality_tokens = modality_tokens.expand(batch_size, modality_tokens.size(1))
                    else:
                        new_modality_tokens = torch.zeros(
                            batch_size, modality_tokens.size(1),
                            dtype=modality_tokens.dtype,
                            device=modality_tokens.device
                        )
                        min_batch = min(batch_size, modality_tokens.size(0))
                        new_modality_tokens[:min_batch] = modality_tokens[:min_batch]
                        modality_tokens = new_modality_tokens
                
                # Now handle sequence length
                modality_tokens = modality_tokens[:, :seq_len]
                if modality_tokens.size(1) < seq_len:
                    padding = torch.zeros(
                        batch_size, seq_len - modality_tokens.size(1),
                        dtype=modality_tokens.dtype,
                        device=modality_tokens.device
                    )
                    modality_tokens = torch.cat([modality_tokens, padding], dim=1)
        
        # Apply the same fix for expression_values
        if expression_values.size(1) != seq_len:
            if expression_values.size(1) == 1:
                expression_values = expression_values.expand(batch_size, seq_len)
            else:
                # Make sure expression_values batch dimension matches gene_tokens
                if expression_values.size(0) != batch_size:
                    if expression_values.size(0) == 1:
                        expression_values = expression_values.expand(batch_size, expression_values.size(1))
                    else:
                        new_expression_values = torch.zeros(
                            batch_size, expression_values.size(1),
                            dtype=expression_values.dtype,
                            device=expression_values.device
                        )
                        min_batch = min(batch_size, expression_values.size(0))
                        new_expression_values[:min_batch] = expression_values[:min_batch]
                        expression_values = new_expression_values
                
                # Now handle sequence length
                expression_values = expression_values[:, :seq_len]
                if expression_values.size(1) < seq_len:
                    padding = torch.zeros(
                        batch_size, seq_len - expression_values.size(1),
                        dtype=expression_values.dtype,
                        device=expression_values.device
                    )
                    expression_values = torch.cat([expression_values, padding], dim=1)

        # Step 1: Embed inputs
        gene_emb = self.gene_embedding(gene_tokens)  # (batch, seq_len, d_model)
        condition_emb = self.condition_embedding(condition_tokens)  # (batch, seq_len, d_model)
        expression_values = expression_values.unsqueeze(-1)  # (batch, seq_len, 1)
        expression_emb = self.expression_fc(expression_values)  # (batch, seq_len, d_model)

        # Step 2: Sum embeddings along axis
        combined_emb = gene_emb + condition_emb + expression_emb  # (batch, seq_len, d_model)

        # Step 3: Add batch and modality embeddings
        batch_emb = self.batch_embedding(batch_ids)  # (batch, seq_len, d_model)
        modality_emb = self.modality_embedding(modality_tokens)  # (batch, seq_len, d_model)
        final_emb = combined_emb + batch_emb + modality_emb  # (batch, seq_len, d_model)

        # Add positional encoding
        pos_enc = self.pos_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)  # (batch, seq_len, d_model)
        final_emb = final_emb + pos_enc

        # Step 4: Transformer decoder with dynamic causal mask
        x = final_emb
        print(f"Input to transformer blocks: {x.shape}")
        all_attn_scores = None
        
        # Create causal mask with correct dimensions
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
        causal_mask = causal_mask.expand(batch_size, self.n_heads, seq_len, seq_len)  # [batch_size, n_heads, seq_len, seq_len]
        
        for i, block in enumerate(self.transformer_blocks):
            x, attn_scores = block(x, causal_mask)
            
            if i < len(self.transformer_blocks) - 1:  # Update mask for next layer if not last layer
                if all_attn_scores is None:
                    all_attn_scores = attn_scores
                else:
                    all_attn_scores = all_attn_scores + attn_scores
                    
                # Update mask dynamically based on attention scores
                if i > 0:  # Skip for the first few layers if desired
                    dyn_mask = self._create_dynamic_causal_mask(seq_len, attn_scores.mean(dim=1), device)
                    causal_mask = dyn_mask  # Already [batch_size, n_heads, seq_len, seq_len]

        # Normalize accumulated attention scores
        if all_attn_scores is not None:
            all_attn_scores = all_attn_scores / len(self.transformer_blocks)

        # Step 5: Output logits
        logits = self.output_layer(x)  # (batch, seq_len, vocab_size)
        
        return logits, all_attn_scores, expression_emb
    
class DynamicMaskedLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.01):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, model_output, targets):
        # Unpack model outputs
        logits, attn_scores, expression_emb = model_output
        
        # logits: (batch, seq_len, vocab_size)
        # targets: (batch, seq_len)
        # attn_scores: (batch, n_heads, seq_len, seq_len)
        # expression_emb: (batch, seq_len, d_model)

        batch_size, seq_len, vocab_size = logits.size()
        
        # Create mask from attention scores
        if attn_scores is None:
            # If no attention scores provided, use a standard causal mask
            mask = torch.tril(torch.ones(batch_size, seq_len, seq_len, device=logits.device))
        else:
            attn_scores = attn_scores.mean(dim=1)  # Average over heads: (batch, seq_len, seq_len)
            # Create mask from attention scores
            threshold = attn_scores.mean()  # Simple threshold for demo
            mask = (attn_scores > threshold).float()
            # Ensure causal property
            causal_mask = torch.tril(torch.ones(batch_size, seq_len, seq_len, device=logits.device))
            mask = mask * causal_mask
        
        # Extract diagonal for token-wise mask: (batch, seq_len)
        token_mask = mask.diagonal(dim1=-2, dim2=-1).float()

        # Cross-entropy loss for tokens
        logits_flat = logits.view(-1, vocab_size)  # (batch * seq_len, vocab_size)
        targets_flat = targets.view(-1)  # (batch * seq_len)
        
        # Calculate losses per token
        ce_losses = self.ce_loss(logits_flat, targets_flat)  # (batch * seq_len)
        ce_losses = ce_losses.view(batch_size, seq_len)  # (batch, seq_len)
        
        # Apply token mask to focus on relevant tokens
        masked_ce_loss = (token_mask * ce_losses).sum() / (token_mask.sum() + 1e-6)

        # Dynamic masking penalty
        if attn_scores is not None:
            attn_max = attn_scores.max(dim=-1)[0]  # Max attention score per position: (batch, seq_len)
            masking_penalty = (1 - token_mask) * (1 - attn_max) ** 2
            masking_loss = self.alpha * masking_penalty.mean()
        else:
            masking_loss = 0.0

        # Expression embedding regularization
        reg_loss = self.beta * torch.norm(expression_emb, p=2, dim=-1).mean()

        # Total loss
        total_loss = masked_ce_loss + masking_loss + reg_loss
        
        return total_loss