from torch.utils.data import DataLoader
import torch

def create_dataloaders(dataset, batch_size=32, shuffle=True, num_workers=2):
    """
    Creates DataLoader for the given dataset.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                    num_workers=num_workers, collate_fn=collate_fn)

def collate_fn(batch):
    """
    Custom collate function to handle varying context sizes and format the batch correctly.
    """
    contexts, targets = zip(*batch)
    
    # Stack context embeddings and targets into batches
    contexts = torch.stack(contexts)  # Shape: (batch_size, context_size, embedding_dim)
    targets = torch.tensor(targets)   # Shape: (batch_size,)
    
    return contexts, targets
