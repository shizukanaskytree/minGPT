from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import torch
import datasets

class TrainDataset:

    def __init__(self, args, train_dataset, data_collator):
        self.args = args
        self.train_dataset = train_dataset
        """
        dataset
        <datasets.iterable_dataset.iterable_dataset.<locals>.TorchIterableDataset object at 0x7f8c068169d0>
        """
        self.data_collator=data_collator


    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `self.train_dataset` does not implement `__len__`, a random sampler (adapted to
        distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size, # 1
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers, # 1
                pin_memory=self.args.dataloader_pin_memory, # from args default.
            )