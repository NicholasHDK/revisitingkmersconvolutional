import torch
import math
import time
import sys
import random
import pickle as pkl
import itertools
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from functools import partial
import argparse
from torch import device
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import re
import os
import csv
import pandas as pd
from torch import Tensor
from typing import List
from torch.nn import functional as F
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Callable
from functools import partial

getitem = []
_collate_fn = []
_load_data_to_loop = []
move_data_to_device = [] 
total_time_to_load_data = []
_process_batch = []
_full_batch_process = []
conv_layer = []
fc_layer = []
calculate_loss = []
backprop = []

# Helper functions
def set_seed(seed: int):
    """
    Set the seed value for reproducibility
    """
    # Set the seed
    random.seed(seed)
    torch.manual_seed(seed)

def log1mexp(x: torch.Tensor):
    """
    Compute `log(1 - exp(-x))` in a numerically stable way for x > 0
    """
    log2 = torch.log(torch.tensor(2.0, dtype=x.dtype, device=x.device))
    return torch.where(
        x < log2, torch.log(-torch.expm1(-x)), torch.log1p(-torch.exp(-x))
    )

class KmerConverter:
    def __init__(self, k):

        # Define the letters, k-mer size, and the base complement
        self.__k = k
        self.__letters = ['A', 'C', 'G', 'T']
        self.__kmer2id = {''.join(kmer): i for i, kmer in enumerate(itertools.product(self.__letters, repeat=self.__k))}
        self.__kmers_num = len(self.__kmer2id)

    def get_k(self):
        return self.__k

    def get_kmers_num(self):
        return self.__kmers_num

    def seq2kmer(self, seq:str, normalized:bool=True):

        # Get the k-mer profile
        kmer2id = [self.__kmer2id[seq[i:i + self.__k]] for i in range(len(seq) - self.__k + 1)]
        kmer_profile = np.bincount(kmer2id, minlength=self.__kmers_num)

        if normalized:
            kmer_profile = kmer_profile / kmer_profile.sum()

        return kmer_profile


class PairDataset(Dataset):
    def __init__(self, file_path, k, neg_sample_per_pos=1000, max_seq_num=0, verbose=True, seed=0):
        """
        PairDataset constructor
        """
        # Set the parameters
        self.__kmers = None # A tensor of shape N x D, storing the k-mer profiles
        self.__indices = None # A tensor of shape L x 2, storing the indices of the positive and negative pairs
        self.__labels = None  # A tensor of shape L, storing the labels of the pairs
        self.__kc = KmerConverter(k)
        self.__neg_sample_per_pos = neg_sample_per_pos

        # Set the seed
        set_seed(seed)

        if verbose:
            print(f"+ Reading the data file.")
            print(f"\t- File path: {file_path}")
            init_time = time.time()

        # Get the number of lines
        num_of_lines = sum(1 for _ in open(file_path, 'r'))

        # If the max_read_num is set, then sample the line numbers to read
        if max_seq_num > 0:
            chosen_lines = random.sample(range(num_of_lines), max_seq_num)
            chosen_lines.sort()
        # Otherwise, read all the lines
        else:
            chosen_lines = list(range(num_of_lines))

        # Read the file
        left_kmer_profiles, right_kmer_profiles = [], []
        # with open(file_path, 'r') as f:
        #     reader = list(csv.reader(f, delimiter=','))
        lines = pd.read_csv(file_path, sep=',', header=None).iloc[chosen_lines].values.tolist()
        for line in lines:
            left_seq, right_seq = line[0], line[1]
            left_kmer_profiles.append(self.__kc.seq2kmer(left_seq))
            right_kmer_profiles.append(self.__kc.seq2kmer(right_seq))

        # Combine the left and right k-mer profiles.
        # The first half of the profiles are the left ones and the second half are the right ones
        left_kmer_profiles = torch.tensor(np.array(left_kmer_profiles))
        right_kmer_profiles = torch.tensor(np.array(right_kmer_profiles))

        self.__kmers = torch.vstack([left_kmer_profiles, right_kmer_profiles]).to(torch.float)

        if verbose:
            print(f"\t- Completed in {time.time() - init_time:.2f} seconds.")
            # print the time information in 2 decimal points
            print(f"\t- The input file contains: {num_of_lines} sequence pairs.")
            print(f"\t- For training, {max_seq_num} sequence pairs will be used.")

        # Construct the indices storing the positive and negative pairs
        if verbose:
            print(f"+ Constructing the indices for positive and negative sample pairs")
            init_time = time.time()

        # Sample 'pos_indices' positive pair indices
        pos_indices = torch.vstack((
            torch.arange(self.__kmers.shape[0]//2, dtype=torch.long),
            torch.arange(self.__kmers.shape[0]//2, dtype=torch.long)+self.__kmers.shape[0]//2
        ))
        # Sample 'neg_sample_per_pos' negative pair indices for each positive pair
        temp_weights = torch.ones((self.__kmers.shape[0],), dtype=torch.float)
        neg_indices = torch.vstack((
            torch.multinomial(temp_weights, pos_indices.shape[1]*self.__neg_sample_per_pos, replacement=True),
            torch.multinomial(temp_weights, pos_indices.shape[1]*self.__neg_sample_per_pos, replacement=True),
        ))
        # Concatenate the positive and negative indices
        self.__indices = torch.hstack((pos_indices, neg_indices))
        self.__labels = torch.hstack((
            torch.ones((pos_indices.shape[1],), dtype=torch.float),
            torch.zeros((neg_indices.shape[1],), dtype=torch.float))
        )

        if verbose:
            print(f"\t- Completed in {time.time() - init_time:.2f} seconds.")
            print(f"\t- Training dataset contains {pos_indices.shape[1]} positive pairs.")
            print(f"\t- Training dataset contains {neg_indices.shape[1]} negative pairs.")

    def __len__(self):
        """
        Return the number of pairs
        """
        return self.__indices.shape[1]

    def __getitem__(self, idx):
        t0 = time.time()
        result = torch.index_select(self.__kmers, dim=0, index=self.__indices[0, idx]), torch.index_select(self.__kmers, dim=0, index=self.__indices[1, idx]), self.__labels[idx]
        t1 = time.time()
        getitem.append(t1 - t0)
        # self.__kmers[self.__indices[0, idx]], self.__kmers[self.__indices[1, idx]], self.__labels[idx]
        return result

# Nich: define convolutional input layer as module
class ConvFeatureExtractor(nn.Module):
    def __init__(self, 
        k: int,
        n_filters: int, 
        device, 
        aggregate_fn: Callable[[Tensor], Tensor] = partial(torch.sum, dim=1)
        ):
        super().__init__()
        self.k = k
        self.aggregate_fn = aggregate_fn
        self.n_filters = n_filters
        # create equivalent kmers as one-hots
        from itertools import product
        self.one_hot_kmer_idcs = torch.tensor(list(product([0,1,2,3], repeat=k))).long() # (4**k, k)
        # create indices to gather from. 
        self.gather_idcs = self.one_hot_kmer_idcs.unsqueeze(0).expand(self.n_filters, -1, -1).unsqueeze(2).to(device) # (filters, 4**k, 1, k)
        # create one_hot equivalents of kmers
        one_hot_kmers = torch.stack([
            torch.nn.functional.one_hot(torch.tensor(indices), num_classes = 4) 
            for indices in list(product([0,1,2,3], repeat=k))
            ]).permute(0,2,1) # (4**k, 4, k)

        if 4**k < n_filters:
            randomized_filters = torch.randn([n_filters - 4**k, 4, k])
            filters = torch.vstack([one_hot_kmers, randomized_filters])
            self.kmer_params = torch.nn.Parameter(filters + torch.randn_like(filters.float()))
        else:
            self.kmer_params = torch.nn.Parameter(one_hot_kmers[n_filters] + torch.randn_like(one_hot_kmers.float())[n_filters]) # (n_filters, 4, k)
        
        #self.kmer_params = torch.nn.Parameter(one_hot_kmers.float(), requires_grad=False) # TODO DELETE
        self.temperature = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True) # TODO CHANGE INITIAL VALUE AND REQUIRES GRAD

    def convolve(self, freq: Tensor):
        """
        Faster convolve, taking advantage of the fact that sequences are one-hot vectors.
        Args:
            freq: kmer frequencies with shape (B, 4**k)
        """
        kmer_params_view = self.kmer_params.unsqueeze(1).expand(-1, 4**self.k, -1, -1) # (n_filters, 4**k, 4, k)
        # torch.gather gets values from self.kmer_params where self.one_hot_kmers is 1, i.e. it is equal to convolution
        matches = torch.gather( 
            kmer_params_view, # (n_filters, 4**k, 4, k)
            2, 
            self.gather_idcs # (n_filters, 4**k, 1, k)
            ) # gather along the onehot dimension, leaving a tensor of shape (filters, 4**k, 1, k)
        matches = matches.squeeze() # (filters, 4**k, k)
        matches = matches.sum(dim=-1) # sum over k dimension (equivalent to the sum in convolution)
        
        matches = torch.softmax(matches/self.temperature, dim=1)
        # this line serves both as a pooling layer (with sum) and to rescale filter outputs by frequencies
        pooled_feature_maps = freq @ matches.T # (batch_size, filters) 
        profile = pooled_feature_maps / pooled_feature_maps.sum(1).unsqueeze(1)
        return profile

    def forward(self, freq: Tensor):
        """frequencies is a tensor of shape (B, 4**k)"""
        y = self.convolve(freq)
        return y

class VIBModel(torch.nn.Module):
    def __init__(self, k:int, n_filters: int, out_dim:int=256, seed:int=0, device='cpu'):
        """
        Initialize the VIB model
        """
        super(VIBModel, self).__init__()

        # Set the parameters
        self.__k = k
        self.__out_dim = out_dim
        # Define the k-mer converter object
        self.__kc = KmerConverter(self.__k)
        self.n_filters = n_filters
        # Set the seed
        set_seed(seed)

        # Nich: define convolutional input layer
        self.conv_feature_extractor = ConvFeatureExtractor(k, n_filters, device)
        
        # Define the layers
        self.linear1_common = torch.nn.Linear(n_filters, 512, dtype=torch.float)
        self.batch1_common = torch.nn.BatchNorm1d(512, dtype=torch.float)
        self.activation1_common = torch.nn.Sigmoid()
        self.dropout1_common = torch.nn.Dropout(0.2)
        self.linear2_mean = torch.nn.Linear(512, self.__out_dim, dtype=torch.float)
        #self.linear2_std = torch.nn.Linear(512, self.__out_dim, dtype=torch.float)

    def encoder(self, kmers: torch.Tensor):
        """
        For given kmers, return the mean and std
        """
        # Nich: extract features
        t0 = time.time()
        feats = self.conv_feature_extractor(kmers)
        t1 = time.time()
        conv_layer.append(t1 - t0)
        h = self.linear1_common(feats)
        h = self.batch1_common(h)
        h = self.activation1_common(h)
        h = self.dropout1_common(h)
        mean = self.linear2_mean(h)
        #std = torch.nn.functional.softplus(self.linear2_std(h)) + 1e-6
        t2 = time.time()
        fc_layer.append(t2 - t1)
        return mean

    def forward(self, left_kmers: torch.Tensor, right_kmers:torch.Tensor):
        """
        Forward pass
        """
        left_mean = self.encoder(left_kmers)
        right_mean = self.encoder(right_kmers)

        return left_mean, right_mean

    def get_k(self):
        """
        Return the k value
        """
        return self.__k

    def get_out_dim(self):
        """
        Return the output dimension
        """
        return self.__out_dim

    def get_seed(self):
        """
        Return the seed value
        """
        return self.__seed

    def seq2emb(self, sequences:list, normalized:bool=True):
        """
        Get the embeddings for the given list of DNA sequences
        """

        kmers = torch.from_numpy(
            np.asarray([self.__kc.seq2kmer(seq, normalized=normalized) for seq in sequences])
        ).to(torch.float)

        with torch.no_grad():
            self.eval()

            if normalized:
                kmers = kmers / kmers.sum(dim=1, keepdim=True)

            means = self.encoder(kmers)

        return means.detach().numpy()

    def save(self, path:str):
        """
        Save the model
        """
        torch.save(
            [{'k': self.get_k(), 'out_dim': self.get_out_dim(), 'n_filters': self.n_filters}, self.state_dict()], path
        )


def loss_func(
        left_mean: torch.Tensor, left_std: torch.Tensor, right_mean: torch.Tensor, right_std: torch.Tensor,
        labels: torch.Tensor, loss_name:str = "vib", include_std:bool = True, samples_num = 8
):
    """
    Definition of the loss functions
    """

    if loss_name == "bern":

        p = torch.exp(-torch.norm(left_mean - right_mean, dim=1)**2)
        return torch.nn.functional.binary_cross_entropy(p, labels, reduction='mean')

    elif loss_name == "vib":

        if include_std:

            epsilon_left = torch.randn((samples_num, left_mean.shape[0], left_mean.shape[1]), device=left_mean.device)
            epsilon_right = torch.randn((samples_num, left_mean.shape[0], left_mean.shape[1]), device=left_mean.device)
            z_i = left_mean + left_std * epsilon_left  # shape (k, batch, d)
            z_j = right_mean + right_std * epsilon_right  # shape (k, batch, d)
            # Compute the distance
            distance = torch.sum((z_i - z_j) ** 2, dim=2)  # shape (k, batch)
            # Get log probabilities, \sum_k exp(-dist) / K (In fact, we don't need the normalization)
            log_p_yij_ri_rj = torch.logsumexp(-distance, dim=0) - torch.log(torch.tensor(samples_num, dtype=torch.float))

        else:

            log_p_yij_ri_rj = -torch.sum((left_mean - right_mean) ** 2, dim=1)

        # Define logits, log(p / (1-p))
        logits = log_p_yij_ri_rj - log1mexp(-log_p_yij_ri_rj)

        return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='mean')

    elif loss_name == "non_stable_vib":

        if include_std:

            epsilon_left = torch.randn((samples_num, left_mean.shape[0], left_mean.shape[1]), device=left_mean.device)
            epsilon_right = torch.randn((samples_num, left_mean.shape[0], left_mean.shape[1]), device=left_mean.device)
            z_left = left_mean + left_std * epsilon_left  # shape (samples, batch, d)
            z_right = right_mean + right_std * epsilon_right  # shape (samples, batch, d)

        else:

            z_left, z_right = left_mean, right_mean

        # Define the success probability
        p = torch.mean(torch.exp(-torch.sum((z_left - z_right) ** 2, dim=-1)), dim=0, keepdim=False)

        return torch.nn.functional.binary_cross_entropy(p, labels, reduction='mean')

    elif loss_name == "vib_without_sampling":
        """
        Computation of the E_p(z_i|r_i)p(z_j|r_j)[p(y_ij|z_i,z_j)] without applying any sampling
        """

        var_left, var_right = left_std**2, right_std**2
        ''' '''
        # Compute the terms log(1+2(s_i^2 + s_j^2))
        log1p_cov = torch.log1p(2*(var_left + var_right))
        # Compute the term m_i - m_j
        mean_squared_diff = (left_mean - right_mean)**2
        # Compute the log expectation
        log_expectation = -0.5*log1p_cov.sum(dim=1) - (mean_squared_diff / torch.exp(log1p_cov)).sum(dim=1)
        # Compute the logits
        logits = log_expectation - log1mexp(-log_expectation)

        return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='mean')

    else:
        raise ValueError(f"Unknown loss function name: {loss_name}")


def train_batch(model, criterion, optimizer, left_kmers: torch.Tensor, right_kmers: torch.Tensor, labels: torch.Tensor):
    # Zero your gradients since PyTorch accumulates gradients on subsequent backward passes.
    optimizer.zero_grad()

    # Make predictions for the current epoch
    left_mean, right_mean = model(left_kmers, right_kmers)

    # Compute the loss and backpropagate
    t0 = time.time()
    batch_loss = criterion(left_mean, 0, right_mean, 0, labels)
    t1 = time.time()
    calculate_loss.append(t1 - t0)
    batch_loss.backward()
    param = model.module.conv_feature_extractor.kmer_params
    #print("Grad None?", param.grad is None)
    #if param.grad is not None:
    #print("Mean grad:", (param - model.module.conv_feature_extractor.kmer_params_copy).abs().mean().item())
    #print("First dense layer:", model.module.linear1_common.weight.grad.abs().mean().item())
    # Update the model parameters
    optimizer.step()
    t2 = time.time()
    backprop.append(t2 - t1)
    return batch_loss

def train_single_epoch(device, model, criterion, optimizer, data_loader):
    epoch_loss = 0.
    i = 0
    t0 = time.time()
    for data in data_loader:
        t1 = time.time()
        left_kmers, right_kmers, labels = data
        left_kmers = left_kmers.reshape(-1, left_kmers.shape[-1]).to(device)
        right_kmers = right_kmers.reshape(-1, right_kmers.shape[-1]).to(device)
        labels = labels.reshape(-1).to(device)
        t2 = time.time()
        move_data_to_device.append(t2-t1)
        total_time_to_load_data.append(t2-t0)

        # Run the training for the current batch
        t3 = time.time()
        batch_loss = train_batch(
            model=model, criterion=criterion, optimizer=optimizer,
            left_kmers=left_kmers, right_kmers=right_kmers, labels=labels
        )
        t4 = time.time()
        _process_batch.append(t4 - t3)
        _full_batch_process.append(t4-t0)
        t0 = time.time()

        # Get the epoch loss for reporting
        epoch_loss += batch_loss

    # Get the average epoch loss
    average_epoch_loss = epoch_loss.item() / len(data_loader)
    return average_epoch_loss

def train(device, distributed, model, criterion, optimizer, data_loader, epoch_num: int, save_every:int, output_path, args):

    # Switch the model to training mode
    model.train()

    ## Start training
    print(f"+ Training started.")
    for current_epoch in range(args.start_epoch, epoch_num):

        # Set the epoch for the sampler
        if distributed:
            data_loader.sampler.set_epoch(current_epoch)

        ## Run the training for the current epoch
        init_time = time.time()
        loss = train_single_epoch(
            device=device, model=model, criterion=criterion, optimizer=optimizer, data_loader=data_loader
        )
        if not distributed or device == 0:
            print(f"\t- Epoch: {current_epoch}/{epoch_num} - Loss: {loss} ({time.time() - init_time:.2f} secs)")

        ## Save the checkpoint if necessary
        if save_every > 0 and (current_epoch + 1) % save_every == 0:
            # Get the folder path of the output file and define the checkpoint path
            checkpoint_path = os.path.join(output_path+f".epoch_{current_epoch+1}.checkpoint")
            if distributed:
                if device == 0:
                    model.module.save(checkpoint_path)
            else:
                model.save(checkpoint_path)


        # # To ensure that all processes have finished the current epoch
        if distributed:
            torch.distributed.barrier()
    if not distributed or device == 0:    
        print("getitem", torch.std_mean(torch.tensor(getitem)))
        print("collate_fn", torch.std_mean(torch.tensor(_collate_fn)))
        print("move_data", torch.std_mean(torch.tensor(move_data_to_device)))
        print("load_data_total", torch.std_mean(torch.tensor(total_time_to_load_data)))
        print("process_batch_total", torch.std_mean(torch.tensor(_process_batch)))
        print("conv_layer", torch.std_mean(torch.tensor(conv_layer)))
        print("fc_layer", torch.std_mean(torch.tensor(fc_layer)))
        print("calc_loss", torch.std_mean(torch.tensor(calculate_loss)))
        print("backprop", torch.std_mean(torch.tensor(backprop)))
        print("full_batch_process", torch.std_mean(torch.tensor(_full_batch_process)))
    print(model.module.conv_feature_extractor.kmer_params[0])
    if distributed:
        if device == 0:
            model.module.save(output_path)
            print(f"Model saved to {output_path}")
    else:
        model.save(output_path)
        print(f"Model saved to {output_path}")
    print(f"\t- Completed (Device {device}).")

def dense_to_onehot(read: Tensor) -> Tensor:
        result = torch.zeros((4, read.shape[-1]), dtype=torch.float)
        result[0, read == 0] = 1
        result[1, read == 1] = 1
        result[2, read == 2] = 1
        result[3, read == 3] = 1
        return result

def batched_dense_to_onehot(reads: torch.int64) -> Tensor:
    """
    Convert a batch of DNA sequences from dense encoding to one-hot encoding.
    
    Args:
        reads (Tensor): A tensor of shape (N, L) containing values in {0, 1, 2, 3}.
    
    Returns:
        Tensor: A one-hot encoded tensor of shape (N, 4, L).
    """
    num_classes = 4
    # One-hot encoding using torch.nn.functional.one_hot, shape: (N, L, 4)
    one_hot = torch.nn.functional.one_hot(reads, num_classes=num_classes)
    # Move the one-hot channel to the correct position: (N, 4, L)
    one_hot = one_hot.permute(0, 2, 1)
    return one_hot

# Nich: define collate_fn to use in dataloader
def collate_fn(batch):
    """
    convert dense representation to onehot
    batch: left reads, right reads, labels
    """
    t0 = time.time()
    left_onehots, right_onehots, labels = torch.vstack([item[0] for item in batch]), torch.vstack([item[1] for item in batch]), torch.vstack([item[2] for item in batch])
    #convert to onehot encoding
    #left_onehots = batched_dense_to_onehot(left_dense).unsqueeze(1) # add channel dimension
    #right_onehots = batched_dense_to_onehot(right_dense).unsqueeze(1)
    t1 = time.time()
    _collate_fn.append(t1 - t0)
    return left_onehots, right_onehots, labels

# Nich: add argument 'args', to pass more values
def main_worker(
        device, world_size, distributed:bool, input_file_path: str, output_path: str, neg_sample_per_pos: int,
        max_seq_num: int, k: int, out_dim: int, lr:float, epoch_num:int, batch_size:int, workers_num, save_every: int,
        loss_name: str, seed: int, args, verbose:bool=True
    ):

    
    ### Initialize the device
    if distributed:
        # Set the environment variables for distributed training if not already set
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "12355"
        print(f"+ MASTER_ADDR: {os.environ['MASTER_ADDR']} and MASTER_PORT: {os.environ['MASTER_PORT']}")
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(backend='nccl', rank=device, world_size=world_size)
    print("Creating data set")
    ### Read the dataset and construct the data loader
    training_dataset = PairDataset(
        file_path=input_file_path, k=k, neg_sample_per_pos=neg_sample_per_pos, max_seq_num=max_seq_num, seed=seed
    )
    
    # Nich: added collate_fn as argument to training_loader
    # Define a DataLoader that iterates through batches of data.
    training_loader = DataLoader(
        training_dataset, batch_size=batch_size, num_workers=workers_num, pin_memory=True,
        sampler=DistributedSampler(training_dataset) if distributed else None,
        shuffle=False if distributed else True#, collate_fncollate_fn
    )

    ### Define the model
    model = VIBModel(k=k, n_filters=args.num_filters, out_dim=args.out_dim, seed=args.seed, device=device)
    
    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model.to(f"cuda:{device}"), device_ids=[device])
    else:
        model.to(device)

    ### Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ### Define the loss function
    criterion = partial(loss_func, loss_name=loss_name, include_std=False)

    ### Train the model
    train(
        device=device, distributed=distributed, model=model, criterion=criterion, optimizer=optimizer,
        data_loader=training_loader, epoch_num=epoch_num, save_every=save_every, output_path=output_path, args=args
    )

    if distributed:
        # Wait for all processes to complete
        torch.distributed.barrier()
        # Release the distributed training resources
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--input', type=str, help='Input sequence file', required=True)
    parser.add_argument('--output_path', type=str, help='Model file path to save', required=True)
    parser.add_argument('--k', type=int, default=4, help='k value')
    parser.add_argument('--out_dim', type=int, default=256, help='dimension value')
    parser.add_argument('--neg_sample_per_pos', type=int, default=1000, help='Negative sample ratio')
    parser.add_argument('--max_seq_num', type=int, default=10000, help='Maximum number of sequences to get from the file')
    parser.add_argument('--epoch', type=int, default=300, help='Epoch number')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=10000, help='Batch size (0: no batch)')
    parser.add_argument('--save_every', type=int, default=0, help='Save checkpoint every x epochs (0 for no checkpoint)')
    parser.add_argument('--distributed', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu, mps or gpu)')
    parser.add_argument('--workers', type=int, default=2, help='Number of data loading workers')
    parser.add_argument('--loss_name', type=str, default="vib", help='Loss function (bern, vib, non_stable_vib and vib_without_sampling)')
    parser.add_argument('--seed', type=int, default=1, help='Seed for random number generator')
    # Nich: Add code for loading model checkpoints, and add argument to decide the number of filters to use
    # Nich: I use the argument 'k' to decide filter width
    parser.add_argument('--trained_model_path', type=str, default=None, help="Path to trained model to resume training")
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--num_filters', type=int, default=136)
    args = parser.parse_args()

    # Define the world size, i.e. total number of processes participating in the distributed training job
    world_size = None

    # Nich: move all cuda code into main_worker to avoid bugs
    if args.distributed:
        assert torch.cuda.is_available(), "Distributed training requires CUDA"
        nodes_num = 1
        world_size = torch.cuda.device_count() * nodes_num
    else:
        if args.device == "gpu":
            assert torch.cuda.is_available(), "GPU is not available"
            device = torch.device(f"cuda")
        elif args.device == "mps":
            assert torch.backends.mps.is_available(), "MPS is not available"
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    print(f"+ Information")
    if args.distributed:
        print(f"\t- Distributed training is activated with {world_size} GPUs")
    else:
        print(f"\t- No distributed training")
        print(f"\t- Device: {device}")
    print(f"\t- Loss function: {args.loss_name}")
    print(f"\t- Batch size: {args.batch_size}")
    print(f"\t- k: {args.k}")
    print(f"\t- num_filters: {args.num_filters}")

    # Nich: pass 'args' along with arguments
    arguments = (
        world_size, args.distributed, args.input, args.output_path, args.neg_sample_per_pos, args.max_seq_num, args.k,
        args.out_dim, args.lr, args.epoch, args.batch_size, args.workers, args.save_every, args.loss_name,  args.seed, args,
        True
    )
    if args.distributed:
        torch.multiprocessing.spawn(main_worker, nprocs=world_size, join=True, args=arguments)
    else:
        main_worker(*((device,) + arguments))
    print("Completed.")

