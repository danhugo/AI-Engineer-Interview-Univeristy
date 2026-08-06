class Dataset:
    def __len__(self) -> int:
        ...
    
    def __getitem__(self, index: int):
        ...

class DataLoader:
    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = False,
        collate_fn: bool | None = None,
        drop_last: bool = False,
        seed: int | None = None,
        num_workers: int = 0,
        prefetch_factor: int = 2,
    ):
        ...
    
    def __iter__(self) -> Iterator:
        ...

class BackgroundIter:
    def __init__(self):
        ...
    
    def __iter__(self):
        ...

    def __next__(self):
        ...