from typing import Generic, TypeVar, Callable, Iterator
from collections.abc import Mapping
import random
import multiprocessing as mp
import torch

T_co = TypeVar("T_co", covariant=True) # T_co is a type placeholder for read-only output
class Dataset(Generic[T_co]):
    """
    Example:

    class MyDataset(Dataset[tuple[int,int]]):
        def __len__(self) -> int: return 10
        def __getitem__(self, i: int) -> tuple[int, int]:
            return i, 2 * i

    Defining T_co makes returned element type known to type-checker

    Example: x, y = my_dataset[0] # checker knows x: int, y: int
    """
    def __len__(self) -> int: raise NotImplementedError    
    def __getitem__(self, index: int) -> T_co: raise NotImplementedError

def default_collate(samples: list):
    """Turn a list of samples into a batch.

    Realistic example — an image dataset where dataset[i] returns
    (image_tensor, label):

        samples = [
            (torch.tensor([[12.,15.,18.],[20.,22.,25.]]), 3),  # image_0, cat
            (torch.tensor([[30.,33.,36.],[40.,42.,45.]]), 7),  # image_1, dog
            (torch.tensor([[50.,53.,56.],[60.,62.,65.]]), 3),  # image_2, cat
        ]

        default_collate(samples)
        # -> ( image_batch, label_batch ) where
        #    image_batch.shape == (3, 2, 3)   # 3 images stacked, each still 2x3
        #    label_batch        == tensor([3, 7, 3])

    Example where data is dict-like:

       samples = [
            {"input_ids": tensor([1,2,3]), "label": 0, "length": 3},
            {"input_ids": tensor([4,5,6]), "label": 1, "length": 3},
            {"input_ids": tensor([7,8,9]), "label": 1, "length": 3},
        ]

        default_collate(samples) =
        {
            "input_ids": tensor([[1,2,3], [4,5,6], [7,8,9]]),   # (3, 3)
            "label":     tensor([0, 1, 1]),                      # (3,)
            "length":    tensor([3, 3, 3]),                      # (3,)
        }
        
    Dispatch:
      - Tensor (same shape) -> torch.stack along a new batch dim 0
      - Tensor (ragged)     -> list[Tensor]  (can't stack mismatched shapes)
      - int / float         -> torch.tensor  (1D, shape [batch])
      - str / bytes         -> list          (kept as-is, not tensor data)
      - dict                -> recurse per key
      - tuple / list        -> transpose fields (zip(*)) and recurse

    The isinstance(elem, Tensor) check is what stops recursion at a
    tensor leaf. Without it, a 2x3 image would be transposed into 3x2
    and have pixels from different images mixed into one row.
    """
    if not samples:
        return []

    elem = samples[0]

    # --- tensor leaf: stack, do NOT transpose ---
    if isinstance(elem, torch.Tensor):
        if all(e.shape == elem.shape for e in samples):
            return torch.stack(samples, dim=0)   # [batch, *elem.shape]
        return list(samples)                      # ragged -> can't stack

    # --- scalar leaf: wrap into a 1D tensor ---
    if isinstance(elem, (int, float)):
        return torch.tensor(samples)
    if isinstance(elem, (str, bytes)):
        return list(samples)                      # strings aren't tensor data

    # --- dict: collate each field separately, keep keys ---
    if isinstance(elem, Mapping): # Mapping to check for dict-like object
        return {k: default_collate([d[k] for d in samples]) for k in elem}

    # --- field container (tuple/list): transpose and recurse ---
    if isinstance(elem, (tuple, list)):
        return type(elem)(default_collate(g) for g in zip(*samples))

    # --- anything else: return as-is ---
    return list(samples)

class DataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = False,
        collate_fn: Callable | None = None,
        drop_last: bool = False,
        seed: int | None = None,
        num_workers: int = 0,
        prefetch_factor: int = 2,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn or default_collate
        self.drop_last = drop_last
        self._rng = random.Random(seed) if seed is not None else random
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
    
    def __iter__(self) -> Iterator:
        """Yield batches for one epoch.

        INTUITION
        ---------
        num_workers=0 : fetch+collate inline in the main process (sequential).
                        Simple, fine for in-memory datasets.

        num_workers>=1: spawn worker PROCESSES (not threads), each running the
                        fetch+collate loop, and feed batches back through a
                        queue. The main loop pulls prepared batches while the
                        GPU computes on the previous one.

        Why processes, not threads? The bottleneck of real datasets is usually
        CPU work in __getitem__ (image decode, tokenization, augmentation).
        Python's GIL lets only one thread run Python at a time, so threads
        would serialize that CPU work. Processes have separate GILs, so they
        truly run in parallel across cores. This is exactly what torch does.

        The cost of processes: dataset + batches must cross the process
        boundary. OUR simple spawn-based loader re-creates the dataset in
        each worker (one copy per worker) -- fine for small study datasets,
        but would exhaust RAM on a big dataset with many workers.

        Real torch AVOIDS that blowup with three tricks (we don't implement):
          - fork() on Linux: copy-on-write shares the dataset pages; workers
            only copy what they mutate. Read-only datasets stay shared.
          - shared-memory tensors: batches go back through the queue via
            /dev/shm, not pickled copies. No duplication of batch data.
          - persistent_workers: workers live across epochs, so no re-copy
            every epoch.

        So "processes copy the dataset" is true for us on spawn, NOT true for
        torch on Linux. That's the known simplification of this study loader.
        """
        # 1. index list for this epoch
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            self._rng.shuffle(indices)

        # 2. turn indices into a stream of batch-index lists.
        #    Note: we send the INDEX LISTS to workers, not the data. Each
        #    worker fetches its own batches via dataset[i]. Indexes are tiny
        #    and cheap to pickle; the fetched data is what we want parallelized.
        def batch_index_lists():
            for start in range(0, len(indices), self.batch_size):
                bidx = indices[start:start + self.batch_size]
                if self.drop_last and len(bidx) < self.batch_size:
                    break
                yield list(bidx)

        # 3a. no workers -> fetch+collate inline in this process
        if self.num_workers == 0:
            for bidx in batch_index_lists():
                samples = [self.dataset[i] for i in bidx]
                yield self.collate_fn(samples)
            return

        # 3b. workers -> process-based prefetch with backpressure
        buf = max(1, self.prefetch_factor * self.num_workers)
        bg = WorkerPool(
            dataset=self.dataset,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            buffer=buf,
        )
        try:
            bg.start(batch_index_lists())
            for batch in bg:
                yield batch
        finally:
            bg.close()

    def __len__(self) -> int:
        """Number of batches (not examples) in one epoch."""
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size   # ceil


class WorkerPool:
    """A pool of worker PROCESSES that fetch+collate batches in parallel.

    INTUITION
    ---------
    Main process            Workers (separate GILs, separate cores)
    ------------            --------------------------------------
    index lists  --feed-->  worker pulls a batch of indexes
                            worker fetches dataset[i] for each  (CPU work, parallel)
                            worker collates into a batch
    prepared batch <-queue- worker puts the batch
    GPU computes            worker starts the next batch

    - Bounded queue = backpressure: workers block on put() once the buffer is
      full, so a slow GPU can't make them build an unbounded pile of batches.
    - Sentinel = end-of-stream: when index lists run out, we send each worker
      a None; it exits. The last sentinel through the queue tells the main
      loop the epoch is done.
    - Errors surface in the main process: a worker that raises wraps the
      exception and sends it through the queue; __next__ re-raises it so the
      training loop sees the real error, not a silent worker death.
    - close() terminates workers: unlike threads, processes can be killed.
      We join briefly, then terminate any that didn't exit cleanly.
    """

    _SENTINEL = None   # tells a worker "no more batches, please exit"

    def __init__(self, dataset, collate_fn, num_workers: int, buffer: int):
        self._dataset = dataset
        self._collate_fn = collate_fn
        self._num_workers = num_workers
        self._buffer = buffer
        # Two queues, opposite directions:
        #   index_q: main -> workers. Carries tiny index lists (cheap to pickle).
        #   batch_q : workers -> main. Carries the prepared batches, BOUNDED so a
        #             slow GPU can't let workers pile up unbounded batches in RAM
        #             (backpressure: workers block on put() when this is full).
        # NOTE: these are mp.Queue -- batches are PICKLED to cross the process
        # boundary. Real torch uses shared-memory tensors so the batch data is
        # not copied; ours pickles, which is fine for small study batches.
        self._index_q: "mp.Queue" = mp.Queue()
        self._batch_q: "mp.Queue" = mp.Queue(maxsize=buffer)
        self._procs: list = []
        self._closed = False

    def start(self, index_lists) -> None:
        """Spawn workers, then feed them the index lists."""
        for _ in range(self._num_workers):
            p = mp.Process(
                target=_worker_loop,
                args=(self._dataset, self._collate_fn, self._index_q, self._batch_q),
                daemon=True,
            )
            p.start()
            self._procs.append(p)
        # feed all index lists, then one sentinel per worker so each exits
        for bidx in index_lists:
            self._index_q.put(bidx)
        for _ in range(self._num_workers):
            self._index_q.put(self._SENTINEL)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._batch_q.get()            # blocks until a batch is ready
        if item is self._SENTINEL:
            raise StopIteration                # a worker finished and had no more
        if isinstance(item, tuple) and item and item[0] == "__error__":
            raise item[1]                      # re-raise worker error in main
        return item

    def close(self):
        if self._closed:
            return
        self._closed = True
        # join each worker briefly; terminate stragglers (processes can be killed)
        for p in self._procs:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()
        # drain queues so daemons don't linger on blocked puts
        _drain(self._batch_q)
        _drain(self._index_q)


def _worker_loop(dataset, collate_fn, index_q, batch_q):
    """The function each worker process runs.

    Pull index lists -> fetch+collate -> put batches. Repeat until sentinel.
    Runs in its own interpreter with its own GIL, so CPU-heavy __getitem__
    work parallelizes across cores.
    """
    while True:
        bidx = index_q.get()                  # blocks for next index list
        if bidx is None:                      # sentinel -> this worker is done
            batch_q.put(None)                 # tell main we exited
            return
        try:
            samples = [dataset[i] for i in bidx]
            batch_q.put(collate_fn(samples))
        except Exception as e:
            # exceptions don't cross process boundaries cleanly; wrap them
            batch_q.put(("__error__", _PickledError(repr(e))))
            return


class _PickledError:
    """Carry an exception across the process boundary as a string.

    Real exceptions may not pickle, and their class might not exist in the
    main process. We keep repr so the message survives.
    """
    def __init__(self, msg: str):
        self.msg = msg
    def __repr__(self):
        return self.msg


def _drain(q) -> None:
    """Empty a queue without blocking (used during shutdown)."""
    try:
        while True:
            q.get_nowait()
    except Exception:
        pass