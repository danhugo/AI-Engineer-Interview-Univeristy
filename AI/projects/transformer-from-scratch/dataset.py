from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self):
        ...
    
    def __len__(self) -> int:
        ...

    def __getitem__(self, index):
        return super().__getitem__(index)
    
class TranslationCollator:
    def __init__(self):
        pass
    
    def __call__(self):
        ...

