import torch
from torch.utils.data import DataLoader
from scalable import VIBModel, PairDataset, collate_fn

checkpoint = torch.load("./models/scalable_100_k=8_d=256_negsampleperpos=200_maxseq=100000_epoch=50_LR=0.001_batch=8_device=None_loss=vib_without_sampling_seed=1_test9.model.epoch_4.checkpoint",
                        map_location="cpu")
config = checkpoint[0]
state_dict = checkpoint[1]
model = VIBModel(k=config['k'], out_dim=config['out_dim'], n_filters=136)
model.load_state_dict(state_dict)

ds = PairDataset("../dataset/train_100k.csv", 4)
dl = DataLoader(
        ds, batch_size=10, num_workers=0, pin_memory=True,
        collate_fn=collate_fn
    )
left, right, labels = next(iter(dl))
print(model(left, right))
