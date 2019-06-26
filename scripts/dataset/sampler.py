import torch
import torch.distributed as dist


class TestDistributedSampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = (len(self.dataset) // self.num_replicas) + int(
            (len(self.dataset) % self.num_replicas) < self.rank)

    def __iter__(self):
        # deterministically shuffle based on epoch
        indices = torch.arange(0, len(self.dataset))

        # subsample
        indices = indices[self.rank::self.num_replicas]

        return iter(indices)

    def __len__(self):
        return self.num_samples
