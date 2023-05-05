import os
import math
import torch
import pickle
from torch.utils.data import Dataset, DataLoader

amino_dict = {'M': 1, 'K': 2, 'Q': 3, 'T': 4, 'V': 5, 'Y': 6, 'I': 7, 'A': 8, 'S': 9, 'P': 10, 'E': 11, 'H': 12, 'W': 13, 'N': 14, 'L': 15, 'G': 16, 'D': 17, 'R': 18, 'F': 19, 'C': 20}

def padding_sequence(sequence, max_length):
    mask = [1] * len(sequence)
    if len(sequence) < max_length:
        sequence += [0] * (max_length - len(sequence))
        mask += [0] * (max_length - len(mask))
    return sequence, mask

def aminoacid2index(aminoacid):
    return amino_dict[aminoacid]

class NeuproteinSequences():
    def __init__(self, data_root, train_flag=True) -> None:
        super().__init__()
        self.data_root = data_root
        self.data_flag = train_flag
        self.dataset = self.process_dataset(self.data_root)
        self.distance_matrix = self.process_dataset_matrix(self.data_root)
    
    def process_dataset(self, data_root):
        protein_sequences = pickle.load(open(os.path.join(data_root, "seq_list"), 'rb'))
        return protein_sequences
    
    def process_dataset_matrix(self, data_root):
        similarity_matrix = pickle.load(open(os.path.join(data_root, "similarity_matrix_result"), 'rb'))
        distance_matrix = 1 - similarity_matrix / 100.0
        return distance_matrix
    
class NeuproteinDataset(Dataset):
    def __init__(self, dataset, repre_sequences, distance, start_index):
        super().__init__()
        self.dataset = dataset
        self.distance = distance
        self.repre_sequences = repre_sequences
        self.max_length = 512
        self.start_index = start_index
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        sequence = self.dataset[index]
        sequence_pairs = []
        for repre_id, repre_sequence in self.repre_sequences:
            sequence_indexs = [aminoacid2index(aminoacid) for aminoacid in sequence]
            repre_sequence_indexs = [aminoacid2index(aminoacid) for aminoacid in repre_sequence]
            sequence_indexs, sequence_mask = padding_sequence(sequence_indexs, self.max_length)
            repre_sequence_indexs, repre_sequence_mask = padding_sequence(repre_sequence_indexs, self.max_length)
            sequence_pairs.append((sequence_indexs, 
                                   repre_sequence_indexs,
                                   sequence_mask,
                                   repre_sequence_mask, 
                                   index + self.start_index, repre_id,
                                   self.distance[index + self.start_index][repre_id]))
        return sequence_pairs

def batch_collate_fn(batchs):
    sequences, repre_sequences, sequence_masks, repre_sequence_masks, sequence_ids, repre_ids, distances = [], [], [], [], [], [], []
    for batch in batchs:
        for sequence_data in batch:
            sequences.append(sequence_data[0])
            repre_sequences.append(sequence_data[1])
            sequence_masks.append(sequence_data[2])
            repre_sequence_masks.append(sequence_data[3])
            sequence_ids.append(sequence_data[4])
            repre_ids.append(sequence_data[5])
            distances.append(sequence_data[6])
    sequences = torch.LongTensor(sequences)
    repre_sequences = torch.LongTensor(repre_sequences)
    sequence_masks = torch.BoolTensor(sequence_masks)
    repre_sequence_masks = torch.BoolTensor(repre_sequence_masks)
    distances = torch.FloatTensor(distances)
    return sequences, \
           repre_sequences, \
           sequence_masks, \
           repre_sequence_masks, \
           sequence_ids, \
           repre_ids, \
           distances

def get_dataloader(dataset,
                   batch_size,
                   num_workers,
                   shuffle=True):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=batch_collate_fn)
    return dataloader