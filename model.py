import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Model, GlobalAttention

class AnchorModel(nn.Module):
    def __init__(self, hidden_size, anchor_number=5):
        super().__init__()
        self.extractor = Model(hidden_size)
        self.anchor_number = anchor_number
    
    def forward(self, 
                left_inputs, 
                right_inputs,
                left_masks, 
                right_masks):
        batch_size = left_inputs.shape[0] // self.anchor_number
        protein_embeddings, anchor_embeddings = self.extractor(left_inputs, 
                                                               right_inputs,
                                                               left_masks, 
                                                               right_masks)
        assert protein_embeddings.shape[-1] == 160
        protein_with_anchor_distances = torch.norm(protein_embeddings - anchor_embeddings, p=2, dim=1)
        
        # protein_mean_embeddings = protein_with_anchor_distances.unsqueeze(1).view(batch_size, self.anchor_number)
        # protein_mean_embeddings = self.simple_trans(protein_mean_embeddings)
        
        protein_with_anchor_embeddings = protein_embeddings.unsqueeze(1).view(batch_size, self.anchor_number, -1)
        # protein_mean_embeddings = self.pooler(protein_with_anchor_embeddings)
        # protein_mean_embeddings = protein_with_anchor_embeddings[:, 0, :]
        protein_mean_embeddings = torch.mean(protein_with_anchor_embeddings, dim=1).squeeze(1)
        # protein_mean_embeddings = torch.max(protein_with_anchor_embeddings, dim=1)[0].squeeze(1)
        
        anchor_with_protein_embeddings = anchor_embeddings.unsqueeze(1).view(batch_size, self.anchor_number, -1)
        anchor_mean_embeddings = torch.mean(anchor_with_protein_embeddings, dim=0).squeeze(0)
        anchor_with_anchor_distances = torch.norm(anchor_mean_embeddings.unsqueeze(1) - anchor_mean_embeddings.unsqueeze(0), p=2, dim=2)
        
        # calculate the distance matrix between protein and protein, output shape: (batch_size, batch_size)
        protein_with_protein_distances = torch.norm(protein_mean_embeddings.unsqueeze(1) - protein_mean_embeddings.unsqueeze(0), p=2, dim=2)
        
        return protein_with_anchor_distances, protein_with_protein_distances, anchor_with_anchor_distances
    
    @torch.no_grad()
    def forward_features(self, 
                         left_inputs, 
                         right_inputs,
                         left_masks, 
                         right_masks):
        batch_size = left_inputs.shape[0] // self.anchor_number
        protein_embeddings, anchor_embeddings = self.extractor(left_inputs, right_inputs, left_masks, right_masks) # anchor_embeddings
        assert protein_embeddings.shape[-1] == 160

        protein_with_anchor_distances = torch.norm(protein_embeddings - anchor_embeddings, p=2, dim=1)
        
        # protein_mean_embeddings = protein_with_anchor_distances.unsqueeze(1).view(batch_size, self.anchor_number)
        # protein_mean_embeddings = self.simple_trans(protein_mean_embeddings)
        
        protein_with_anchor_embeddings = protein_embeddings.unsqueeze(1).view(batch_size, self.anchor_number, -1)
        # protein_mean_embeddings = self.pooler(protein_with_anchor_embeddings)
        # protein_mean_embeddings = protein_with_anchor_embeddings[:, 0, :]
        protein_mean_embeddings = torch.mean(protein_with_anchor_embeddings, dim=1).squeeze(1)
        # protein_mean_embeddings = torch.max(protein_with_anchor_embeddings, dim=1)[0].squeeze(1)
        
        return protein_mean_embeddings, protein_with_anchor_distances