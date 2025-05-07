import torch
import random
import itertools
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch import nn, Tensor
import re
import os
from torch.nn import functional as F
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '12355')
    dist.init_process_group("nccl", rank=rank, init_method="env://", world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def set_seed(seed):
    # Set the seed
    random.seed(seed)
    torch.manual_seed(seed)


class PairDataset(Dataset):
    def __init__(self, file_path, transform_func, neg_sample_per_pos=1000, max_read_num=0, verbose=True, seed=0):

        # Set the parameters
        self.__both_kmer_profiles = None
        self.__transform_func = self.dense_encoding
        self.__neg_sample_per_pos = neg_sample_per_pos
        self.__seed = seed
        # Set the seed
        set_seed(seed)

        # Get the number of lines
        with open(file_path, 'r') as f:
            lines_num = sum(1 for _ in f)
        # If the max_read_num is set, then sample the line number to read
        if max_read_num > 0:
            chosen_lines = random.sample(range(lines_num), max_read_num)
            chosen_lines.sort()

        # Read the file
        chosen_line_idx = 0
        left_kmer_profiles, right_kmer_profiles = [], []
        with open(file_path, 'r') as f:
            for current_line_idx, line in enumerate(f):

                if max_read_num > 0:
                    if chosen_line_idx == len(chosen_lines):
                        break

                    if current_line_idx != chosen_lines[chosen_line_idx]:
                        continue
                    else:
                        chosen_line_idx += 1

                # Remove the newline character and commas
                left_read, right_read = line.strip().split(',')
                left_kmer_profiles.append(self.__transform_func(left_read))
                right_kmer_profiles.append(self.__transform_func(right_read))
        max_profile_len = max([profile.shape[-1] for profile in left_kmer_profiles + right_kmer_profiles])
        left_kmer_profiles = torch.stack([F.pad(x, (-1, max_profile_len - x.shape[-1])) for x in left_kmer_profiles])
        right_kmer_profiles = torch.stack([F.pad(x, (-1, max_profile_len - x.shape[-1])) for x in right_kmer_profiles])
        
        # Combine the left and right k-mer profiles
        self.__both_kmer_profiles = torch.vstack([left_kmer_profiles, right_kmer_profiles])

        if verbose:
            print(f"The data file was read successfully!")
            print(f"\t+ Total number of read pairs: {lines_num}")
            if max_read_num > 0:
                print(f"\t+ Number of read pairs used: {max_read_num}")

        # Temporary variables
        self.__ones = torch.ones((len(self.__both_kmer_profiles),))

    def dense_encoding(self, read: str) -> Tensor:
        seq = np.array(read, dtype=np.bytes_).reshape(1, -1)
        seq = seq.view(np.uint8).squeeze()
        result = torch.zeros(seq.shape[-1], dtype=torch.uint8)
        result[seq == 65] = 0
        result[seq == 67] = 1
        result[seq == 71] = 2
        result[seq == 84] = 3
        return result
    
    def dense_to_onehot(self, read: Tensor) -> Tensor:
        result = torch.zeros((4, read.shape[-1]), dtype=torch.float)
        result[0, read == 0] = 1
        result[1, read == 1] = 1
        result[2, read == 2] = 1
        result[3, read == 3] = 1
        return result

    def batched_dense_to_onehot(self, reads: Tensor) -> Tensor:
        """
        Convert a batch of DNA sequences from dense encoding to one-hot encoding.
        
        Args:
            reads (Tensor): A tensor of shape (N, L) containing values in {0, 1, 2, 3, -1}.
        
        Returns:
            Tensor: A one-hot encoded tensor of shape (N, 4, L).
        """
        num_classes = 4
        # Handle unknown (-1) by replacing it with 0 temporarily (it will be masked later)
        masked_reads = torch.where(reads < 0, torch.tensor(0, dtype=reads.dtype, device=reads.device), reads).to(torch.int64)
        # One-hot encoding using torch.nn.functional.one_hot, shape: (N, L, 4)
        one_hot = torch.nn.functional.one_hot(masked_reads, num_classes=num_classes).to(torch.float32)
        # Move the one-hot channel to the correct position: (N, 4, L)
        one_hot = one_hot.permute(0, 2, 1)
        # Set unknown values (-1 in input) to all zeros in the one-hot encoding
        one_hot[(reads < 0).unsqueeze(1).expand_as(one_hot)] = 0

        return one_hot

    def __len__(self):
        return len(self.__both_kmer_profiles) // 2

    def __getitem__(self, idx):

        # Sample negative sample_indices
        negative_sample_indices = torch.multinomial(
            self.__ones, replacement=True, num_samples=2*self.__neg_sample_per_pos
        )

        # Define the positive and negative k-mer profile pairs
        left_kmer_profiles = torch.concatenate((
            self.dense_to_onehot(self.__both_kmer_profiles[idx]).unsqueeze(0),
            self.batched_dense_to_onehot(self.__both_kmer_profiles[negative_sample_indices[:self.__neg_sample_per_pos]])
        ))
        right_kmer_profiles = torch.concatenate((
            self.dense_to_onehot(self.__both_kmer_profiles[idx+self.__len__()]).unsqueeze(0),
            self.batched_dense_to_onehot(self.__both_kmer_profiles[negative_sample_indices[self.__neg_sample_per_pos:]])
        ))
        # Define the labels
        labels = torch.tensor([1] + [0] * self.__neg_sample_per_pos, dtype=torch.float)
        return left_kmer_profiles.unsqueeze(1), right_kmer_profiles.unsqueeze(1), labels

class ConvFeatureExtractor(nn.Module):
    def __init__(self, n_filters: int, k: int):
        super().__init__()
        self.n_filters = n_filters
        self.k = k
        self.filters = nn.Conv2d(1, n_filters, kernel_size=(4, k), device="cuda")
        
    def forward(self, X: Tensor):
        """X is a Tensor of shape (B, 1, 4, L)"""
        y = self.filters(X)
        y = y.mean(dim=-1)
        y = y.squeeze(-2) # remove channel dimension
        return y

class NonLinearModel(torch.nn.Module):
    def __init__(self, inputs, dim=256, seed=0):
        super(NonLinearModel, self).__init__()
        self.inputs=inputs
        self.__dim=dim

        # Set the seed
        set_seed(seed)

        # Define the layers
        self.linear1 = torch.nn.Linear(self.inputs, 512, dtype=torch.float)
        self.batch1 = torch.nn.BatchNorm1d(512, dtype=torch.float)
        self.activation1 = torch.nn.Sigmoid()
        self.dropout1 = torch.nn.Dropout(0.2)
        self.linear2 = torch.nn.Linear(512, self.__dim, dtype=torch.float)

        self.bce_loss = torch.nn.BCELoss()

    def encoder(self, kmer_profile):

        output = self.linear1(kmer_profile)
        output = self.batch1(output)
        output = self.activation1(output)
        output = self.dropout1(output)
        output = self.linear2(output)

        return output

    def forward(self, left_kmer_profile, right_kmer_profile):
        left_output = self.encoder(left_kmer_profile)
        right_output = self.encoder(right_kmer_profile)

        return left_output, right_output

class ConvNonLinear(nn.Module):
    """
    Connects a feature extractor input layer to a FC NN.
    """
    def __init__(self, feature_extractor: ConvFeatureExtractor, fc: NonLinearModel):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.fc = fc

    def forward(self, left_X: Tensor, right_X: Tensor):
        left_features, right_features = self.features(left_X), self.features(right_X)
        return self.encoder(left_features), self.encoder(right_features)
    
    def features(self, input):
        return self.feature_extractor(input)
    
    def encoder(self, features):
        return self.fc.encoder(features)


def loss_func(left_embeddings, right_embeddings, labels, name="bern"):

    if name == "bern":
        p = torch.exp(-torch.norm(left_embeddings - right_embeddings, p=2, dim=1)**2 )

        return torch.nn.functional.binary_cross_entropy(p, labels, reduction='mean')

    elif name == "poisson":

        log_lambda = -torch.norm(left_embeddings - right_embeddings, p=2, dim=1)**2

        return torch.mean(-(labels * log_lambda) + torch.exp(log_lambda))

    elif name == "hinge":

        d = torch.norm(left_embeddings - right_embeddings, p=2, dim=1)
        return torch.mean(labels * (d**2) + (1 - labels) * torch.nn.functional.relu(1 - d)**2)

    else:

        raise ValueError(f"Unknown loss function: {name}")

def collate_fn(batch):
        return torch.vstack([item[0] for item in batch]), torch.vstack([item[1] for item in batch]), torch.vstack([item[2] for item in batch])


def single_epoch(model, loss_func, optimizer, training_loader, loss_name="bern", device=None):
    #i=0
    epoch_loss = 0.
    for data in training_loader:
        #print(i/100000)
        #i += 1
        left_kmer_profile, right_kmer_profile, labels = data
        # Zero your gradients since PyTorch accumulates gradients on subsequent backward passes.
        optimizer.zero_grad()
        left_kmer_profile = left_kmer_profile.to(device).squeeze(0)
        right_kmer_profile = right_kmer_profile.to(device).squeeze(0)
        labels = labels.reshape(-1).to(device)

        # Make predictions for the current epoch
        left_output, right_output = model(left_kmer_profile, right_kmer_profile)

        # Compute the loss and backpropagate
        batch_loss = loss_func(left_output, right_output, labels, name=loss_name)
        batch_loss.backward()

        # Update the model parameters
        optimizer.step()

        # Get the epoch loss for reporting
        epoch_loss += batch_loss.item()
        del batch_loss, left_kmer_profile, right_kmer_profile, labels, left_output, right_output
        torch.cuda.empty_cache()

    return epoch_loss / len(training_loader)


def run(rank, world_size, learning_rate, epoch_num, loss_name="bern", model_save_path=None, loss_file_path=None, checkpoint=0, verbose=True, args=None):
    print(f"rank: {rank}, world_size: {world_size}")
    if rank == 0:
        print("Using", torch.cuda.device_count(), "GPUs")
    setup(rank, world_size)
    # Read the dataset
    training_dataset = PairDataset(
        args.input, None, args.neg_sample_per_pos
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(training_dataset)

        # Define the training data loader
    training_loader = DataLoader(
        training_dataset, batch_size=args.batch_size if args.batch_size else len(training_dataset),
        num_workers=10, sampler=train_sampler, collate_fn=collate_fn
    )
    # Define the model
    dist.barrier()
    device = torch.device(f'cuda:{rank}')
    if args.trained_model_path is not None:
        checkpoint_model = torch.load(args.trained_model_path, map_location="cuda:0")

        # Rebuild components
        feature_extractor = ConvFeatureExtractor(args.inputs, args.k).to(device)
        encoder = NonLinearModel(
            inputs=args.inputs, dim=args.dim, seed=args.seed
        ).to(device)

        model = ConvNonLinear(feature_extractor, encoder).to(device)

        # Restore weights
        model.feature_extractor.load_state_dict(checkpoint_model['feature_extractor_state_dict'])
        model.fc.load_state_dict(checkpoint_model['fc_state_dict'])
        model.load_state_dict(checkpoint_model['model_state_dict'])
    else:
        feature_extractor = ConvFeatureExtractor(args.inputs, args.k).to(device) # 256 is the number of features used in revisiting kmers original code
        encoder = NonLinearModel(
            inputs=args.inputs, dim=args.dim, seed=args.seed
        ).to(device)
        model = ConvNonLinear(feature_extractor, encoder).to(device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if loss_file_path is not None:
        writer = SummaryWriter(loss_file_path)
    if rank == 0:
        print("Training has just started.")

    for epoch in range(args.start_epoch, epoch_num):
        if rank == 0:
            print(f"\t+ Epoch {epoch + 1}.")
        model.train()
        train_sampler.set_epoch(epoch)
        # Make sure gradient tracking is on, and do a pass over the data
        avg_loss = single_epoch(model, loss_func, optimizer, training_loader, loss_name, rank)

        if rank == 0:
            print(f"Epoch {epoch + 1}, Training Loss: {avg_loss}")

        if loss_file_path is not None:
            writer.add_scalar('Loss/train', avg_loss, epoch + 1)
            writer.flush()

        if rank == 0 and (epoch + 1) % checkpoint == 0:  # Only save from rank 0 to avoid multiple writes
            save_path = f"model{args.inputs}_{args.k}_checkpoint{epoch}.pt"
            torch.save({
                'feature_extractor_state_dict': model.feature_extractor.state_dict(),
                'fc_state_dict': model.fc.state_dict(),
                'model_state_dict': model.state_dict(),
                'args': args  # optionally store config for reproducibility
            }, save_path)

    writer.close()
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate clustering')
    parser.add_argument('--input', type=str, help='Input sequence file')
    parser.add_argument('--k', type=int, default=2, help='k value')
    parser.add_argument('--dim', type=int, default=256, help='dimension value')
    parser.add_argument('--neg_sample_per_pos', type=int, default=1000, help='Negative sample ratio')
    parser.add_argument('--max_read_num', type=int, default=10000, help='Maximum number of reads to get from the file')
    parser.add_argument('--epoch', type=int, default=1000, help='Epoch number')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=0, help='Batch size (0: no batch)')
    parser.add_argument('--device', type=str, default="cpu", help='Device (cpu or cuda)')
    parser.add_argument('--workers_num', type=int, default=1, help='Number of workers for data loader')
    parser.add_argument('--loss_name', type=str, default="bern", help='Loss function (bern, poisson, hinge)')
    parser.add_argument('--output', type=str, help='Output file')
    parser.add_argument('--seed', type=int, default=26042024, help='Seed for random number generator')
    parser.add_argument('--checkpoint', type=int, default=0, help='Save the model for every checkpoint epoch')
    parser.add_argument('--world_size', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--trained_model_path', type=str, default=None, help='Path to the trained model, in case you want to continue training from a checkpoint')
    parser.add_argument('--inputs', type=int, default=None, help='Number of filters')
    parser.add_argument('--start_epoch', type=int, default=0, help='For resuming training')
    args = parser.parse_args()

    # Run the model
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print("starting..")
    print(f"world_size: {world_size}")
    torch.manual_seed(args.seed)
    run(
        rank,
        world_size,
        args.lr,
        args.epoch,
        args.loss_name,
        args.output,
        args.output + ".loss",
        args.checkpoint,
        True,
        args
    )
