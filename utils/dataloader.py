# coding : utf-8
# Author : yuxiang Zeng
import platform
import multiprocessing

from torch.utils.data import DataLoader


def get_dataloaders(train_set, valid_set, test_set, args):
    max_workers = multiprocessing.cpu_count()
    # max_workers = 1

    train_loader = DataLoader(
        train_set,
        batch_size=args.bs,
        drop_last=False,
        shuffle=True,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        # num_workers=max_workers,
        # prefetch_factor=4
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=4096,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        # num_workers=max_workers,
        # prefetch_factor=4
    )
    test_loader = DataLoader(
        test_set,
        batch_size=4096,  # 8192
        drop_last=False,
        shuffle=False,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        # num_workers=max_workers,
        # prefetch_factor=4
    )

    return train_loader, valid_loader, test_loader


def custom_collate_fn(batch):
    from torch.utils.data.dataloader import default_collate
    return default_collate(batch)
