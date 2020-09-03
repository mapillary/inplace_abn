import torch
import torch.distributed as distributed
import torch.nn as nn


def active_group(active: bool):
    """Initialize a distributed group where each process can independently decide whether to participate or not

    Args:
        active: Whether this process will be active in the group or not

    Returns:
        group: A distributed group containing all processes that passed `active=True`,
            or `None` if all passed `False`
    """
    world_size = distributed.get_world_size()
    rank = distributed.get_rank()

    # Check if cache is initialized, add WORLD and None to it
    if not hasattr(active_group, "__cache__"):
        active_group.__cache__ = {
            frozenset(range(world_size)): distributed.group.WORLD,
            frozenset(): None,
        }

    # Gather active status from all workers
    active = torch.tensor(
        rank if active else -1, dtype=torch.long, device=torch.cuda.current_device()
    )
    active_workers = torch.empty(
        world_size, dtype=torch.long, device=torch.cuda.current_device()
    )
    distributed.all_gather(list(active_workers.unbind(0)), active)

    # Create and cache group if it doesn't exist yet
    active_workers = frozenset(int(i) for i in active_workers.tolist() if i != -1)
    if active_workers not in active_group.__cache__:
        group = distributed.new_group(list(active_workers))
        active_group.__cache__[active_workers] = group

    return active_group.__cache__[active_workers]


def set_active_group(module: nn.Module, group):
    """Scan all submodules, passing a distributed group to all those that implement `set_group`"""

    def _set_group(m):
        if hasattr(m, "set_group"):
            m.set_group(group)

    module.apply(_set_group)
