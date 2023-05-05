import os
import sys
import time
import torch
import pickle
import argparse
import numpy as np
from loss import LossFunc
from model import AnchorModel
from dataloader import NeuproteinSequences, NeuproteinDataset, get_dataloader
from utils import get_repre_sequences, set_seeds, get_distance_matrix_with_postprocess, get_distance_matrix_from_embeddings, get_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--seed', default=2012, type=int, help='random seed')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--skip_train', action='store_false', help='train flag')
    parser.add_argument('--eval_train', action='store_true', help='eval train dataset flag')
    parser.add_argument('--dataroot', default='/mnt/sda/czh/Neuprotein/5000_needle_512', type=str, help='path to dataset')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--hidden_size', default=160, type=int, help='hidden size')
    parser.add_argument('--num_workers', default=4, type=int, help='num workers')
    parser.add_argument('--epochs', default=100, type=int, help='epochs')
    parser.add_argument('--eval_epochs', default=5, type=int, help='eval epochs')
    parser.add_argument('--t_max', default=10, type=int, help='t_max')
    parser.add_argument('--repre_k', default=5, type=int, help='number of representative sequences')
    parser.add_argument('--evaluate_topks', default=[1, 5, 10, 50, 100, 500], type=list, help='evaluate topks')
    parser.add_argument('--train_number', default=3000, type=int, help='number of training sequences')
    parser.add_argument('--query_number', default=500, type=int, help='number of query sequences')
    args = parser.parse_args()
    
    # set random seed
    topks = args.evaluate_topks
    set_seeds(args.seed)
    
    # load dataset
    NSDataset = NeuproteinSequences(args.dataroot)
    train_sequences, test_sequences = NSDataset.dataset[:args.train_number], NSDataset.dataset[args.train_number:]
    train_distance_matrixs = NSDataset.distance_matrix[:args.train_number,:args.train_number]
    test_distance_matrixs = NSDataset.distance_matrix[args.train_number:,args.train_number:]
    
    # Chonsen the most reprensative sequences
    # if not os.path.exists("./cache/repre_sequences"):        
    repre_sequence_ids = get_repre_sequences(train_distance_matrixs, args.repre_k, random_state=args.seed)
    repre_sequences = [[repre_id, train_sequences[repre_id]] for repre_id in repre_sequence_ids]
    pickle.dump(repre_sequences, open("./cache/repre_sequences", 'wb'))
    # else:
    #     repre_sequences = pickle.load(open("./cache/repre_sequences", 'rb'))
    
    # make dataloader
    train_dataset = NeuproteinDataset(train_sequences, repre_sequences, NSDataset.distance_matrix, 0)
    test_dataset = NeuproteinDataset(test_sequences, repre_sequences, NSDataset.distance_matrix, args.train_number)
    
    train_dataloader = get_dataloader(train_dataset, args.batch_size, args.num_workers)
    test_dataloader = get_dataloader(test_dataset, args.batch_size, args.num_workers, shuffle=False)
    
    # make model and train
    model = AnchorModel(args.hidden_size, args.repre_k)
    model.cuda()
    model = model.train()
    loss_function = LossFunc(args.repre_k)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max, eta_min=0.00001)
    
    # init metrics
    mae = 9999
    best_metrics = {'top%d' % topk: 0 for topk in topks}
    
    if args.eval_train:
        print("eval train dataset")
        time.sleep(2)
    
    for epoch in range(args.epochs):
        if args.skip_train:
            model = model.train()
            for (sequences,  
                re_sequences,
                sequence_masks, 
                repre_sequence_masks, 
                sequence_ids, repre_ids, 
                distances) in train_dataloader:
                sequences = sequences.cuda()
                re_sequences = re_sequences.cuda()
                sequence_masks = sequence_masks.cuda()
                repre_sequence_masks = repre_sequence_masks.cuda()
                protein_with_anchor_target_distance = distances.cuda()
                protein_ids = [char for seen, char in enumerate(sequence_ids) if char not in sequence_ids[:seen]]
                protein_with_protein_target_distance = get_distance_matrix_with_postprocess(NSDataset.distance_matrix, protein_ids, protein_ids)
                protein_with_protein_target_distance = protein_with_protein_target_distance.cuda()
                anchor_with_anchor_target_distance = get_distance_matrix_with_postprocess(NSDataset.distance_matrix, repre_sequence_ids, repre_sequence_ids)
                anchor_with_anchor_target_distance = anchor_with_anchor_target_distance.cuda()
                # print(anchor_with_anchor_target_distance)
                protein_with_anchor_predict_distance, protein_with_protein_preidct_distance, anchor_with_anchor_distances = model(sequences, re_sequences, sequence_masks, repre_sequence_masks)
                
                # calculate loss
                loss = loss_function(protein_with_anchor_target_distance, 
                                    protein_with_protein_target_distance,
                                    anchor_with_anchor_target_distance,
                                    protein_with_anchor_predict_distance, 
                                    protein_with_protein_preidct_distance,
                                    anchor_with_anchor_distances)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                print("epoch: {}, loss: {}, metrics: {}, mae: {}".format(epoch, loss, best_metrics, mae), end="")
                # [protein_with_anchor_predict_distance[:args.repre_k].detach().cpu().numpy(), 
                # protein_with_anchor_target_distance[:args.repre_k].detach().cpu().numpy()]
                os.system('clear')
                sys.stdout.flush()
            scheduler.step()
        
        if args.eval_train:
            test_dataset = NeuproteinDataset(train_sequences, repre_sequences, NSDataset.distance_matrix, 0)
            test_dataloader = get_dataloader(test_dataset, args.batch_size, args.num_workers, shuffle=False)
            test_distance_matrixs = train_distance_matrixs
            test_sequences = train_sequences
        
        model = model.eval()
        if epoch % args.eval_epochs == 0:
            model = model.eval()
            with torch.no_grad():
                target_distances = []
                predict_distances = []
                predict_protein_embeddings = []
                for (sequences,  
                    re_sequences,
                    sequence_masks, 
                    repre_sequence_masks, 
                    sequence_ids, repre_ids, 
                    distances) in test_dataloader:
                    sequences = sequences.cuda()
                    re_sequences = re_sequences.cuda()
                    sequence_masks = sequence_masks.cuda()
                    repre_sequence_masks = repre_sequence_masks.cuda()
                    protein_embeddings, protein_with_anchor_distances = model.forward_features(sequences, re_sequences, sequence_masks, repre_sequence_masks)
                    predict_protein_embeddings.append(protein_embeddings)
                    
                    target_distances.append(distances)
                    predict_distances.append(protein_with_anchor_distances)
                
                target_distances = torch.cat(target_distances, dim=0).cpu()
                predict_distances = torch.cat(predict_distances, dim=0).cpu()
                
                mae = np.abs(target_distances - predict_distances).view(-1, args.repre_k).sum(dim=1).mean().item()
                    
                predict_protein_embeddings = torch.cat(predict_protein_embeddings, dim=0)
                
                query_protein_embeddings = predict_protein_embeddings[:args.query_number].cpu().numpy()
                gallery_protein_embeddings = predict_protein_embeddings[args.query_number:].cpu().numpy()
                
                predict_distance_matrix = get_distance_matrix_from_embeddings(query_protein_embeddings, gallery_protein_embeddings)
                target_distance_matrix = test_distance_matrixs[:args.query_number, args.query_number:]
                
                # calculate metrics
                metrics = get_metrics(predict_distance_matrix, target_distance_matrix, topks=topks)
                
                # update best metrics
                update_model_flag = 0
                for topk in topks:
                    if metrics['top%d' % topk] > best_metrics['top%d' % topk]:
                        update_model_flag += 1
                if update_model_flag >= 3: 
                    best_metrics = metrics
                    torch.save(model.state_dict(), "./cache/best_model.pth".format(epoch))
                pickle.dump(target_distances.numpy(), open("./cache/target_distances", 'wb'))
                pickle.dump(predict_distances.numpy(), open("./cache/predict_distances", 'wb'))
                pickle.dump(target_distance_matrix, open("./cache/target_distance_matrix", 'wb'))
                pickle.dump(predict_distance_matrix, open("./cache/predict_distance_matrix", 'wb'))
                pickle.dump(test_sequences, open("./cache/test_sequences", 'wb'))
                pickle.dump(query_protein_embeddings, open("./cache/query_protein_embeddings", 'wb'))
                pickle.dump(gallery_protein_embeddings, open("./cache/gallery_protein_embeddings", 'wb'))
                
            # torch.save(model.state_dict(), "./cache/model_{}".format(epoch))
            
# Original Result: metrics: {'top1': 0.16, 'top5': 0.19, 'top10': 0.2, 'top50': 0.23, 'top100': 0.28, 'top500': 0.42}, mae: 0.063
# With Anchor Loss: metrics: {'top1': 0.16, 'top5': 0.17, 'top10': 0.18, 'top50': 0.24, 'top100': 0.29, 'top500': 0.44}, mae: 0.121