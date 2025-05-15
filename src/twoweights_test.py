import torch
from torch.utils.data import DataLoader
from conv_nonlinear import ConvFeatureExtractor, NonLinearModel, ConvNonLinear, PairDataset, collate_fn

trained_model_path = "./cps/model_checkpoint25.pt"
checkpoint_model = torch.load(trained_model_path, map_location="cuda")

# Rebuild components
feature_extractor = ConvFeatureExtractor(256, 4).to("cuda")
encoder = NonLinearModel(
    inputs=256, dim=256, seed=0
).to("cuda")

model = ConvNonLinear(feature_extractor, encoder).to("cuda")

# Restore weights
model.feature_extractor.load_state_dict(checkpoint_model['feature_extractor_state_dict'])
model.fc.load_state_dict(checkpoint_model['fc_state_dict'])
model.load_state_dict(checkpoint_model['model_state_dict'])

#ds = PairDataset("../dataset/train_100k.csv", 4)
#dl = DataLoader(
#        ds, batch_size=10, num_workers=0, pin_memory=True,
#        collate_fn=collate_fn
#    )
#left, right, labels = next(iter(dl))
print(model.feature_extractor.filters.weight.data)
