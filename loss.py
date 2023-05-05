import torch
import random
import torch.nn as nn
import torch.nn.functional as F

class TripletMarginLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin
        self.lossfunc = nn.SoftMarginLoss

    def get_pos_neg_index(self, target_distances):
        # target_distances shape: (batch_size, batch_size)
        batch_size = target_distances.shape[0]
        pos_index = torch.zeros(batch_size).cuda()
        neg_index = torch.zeros(batch_size).cuda()
        for i in range(batch_size):
            pi, pj = random.sample(range(batch_size), 2)
            while pi == i or pj == i:
                pi, pj = random.sample(range(batch_size), 2)
            if target_distances[i][pi] > target_distances[i][pj]:
                pos_index[i] = pi
                neg_index[i] = pj
            else:
                pos_index[i] = pj
                neg_index[i] = pi
        return pos_index, neg_index

    def forward(self, predict_distances, target_distances):
        pos_index, neg_index = self.get_pos_neg_index(target_distances)
        pos_distances = torch.gather(predict_distances, 1, pos_index.unsqueeze(1).long())
        neg_distances = torch.gather(predict_distances, 1, neg_index.unsqueeze(1).long())
        pos_target_distances = torch.gather(target_distances, 1, pos_index.unsqueeze(1).long())
        neg_target_distances = torch.gather(target_distances, 1, neg_index.unsqueeze(1).long())
        threshold = neg_target_distances - pos_target_distances
        loss = torch.relu(pos_distances - neg_distances + threshold).mean()
        return loss

class LossFunc(nn.Module):
    def __init__(self, anchor_number):
        super().__init__()
        self.regression_loss = nn.L1Loss(reduction='none')
        self.triplet = TripletMarginLoss(margin=1.0)
        self.anchor_number = anchor_number
    
    def extract_non_diagonal_elements(self, matrix):
        # Create an identity matrix of the same size as the input matrix
        identity_matrix = torch.eye(matrix.size(0), matrix.size(1))

        # Create a boolean mask with 'True' for non-diagonal elements and 'False' for diagonal elements
        mask = identity_matrix == 0

        # Extract non-diagonal elements using the mask
        non_diagonal_elements = matrix[mask]

        return non_diagonal_elements
    
    def forward(self, 
                protein_with_anchor_targets, 
                protein_with_protein_targets, 
                anchor_with_anchor_targets,
                protein_with_anchor_predicts, 
                protein_with_protein_predicts,
                anchor_with_anchor_predicts, 
                alpha=1.0,
                beta=1.0,
                gamma=1.0,
                theta=1.0):
        # calculate the anchor loss 
        protein_anchor_loss = self.regression_loss(protein_with_anchor_predicts, protein_with_anchor_targets)
        protein_anchor_loss = protein_anchor_loss.view(-1, self.anchor_number).sum(dim=1).mean()
        # calculate the protein loss
        protein_protein_loss = self.regression_loss(protein_with_protein_predicts, protein_with_protein_targets)
        protein_protein_loss = self.extract_non_diagonal_elements(protein_protein_loss).mean()
        # calculate the triplet loss
        metric_loss = self.triplet(protein_with_protein_predicts, protein_with_protein_targets)
        # calculate the anchor with anchor loss
        anchor_anchor_loss = self.regression_loss(anchor_with_anchor_predicts, anchor_with_anchor_targets)
        anchor_anchor_loss = self.extract_non_diagonal_elements(anchor_anchor_loss).mean()
        
        total_loss = alpha * protein_anchor_loss + beta * protein_protein_loss + gamma * metric_loss + theta * anchor_anchor_loss 
        return total_loss