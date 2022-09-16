from torch.utils.data import Dataset, DataLoader


class ComposedDataset(Dataset):
    def __init__(self, dst_a, dst_b):
        super().__init__()
        self.dst_a = dst_a
        self.dst_b = dst_b

    def __len__(self):
        return max(len(self.dst_a), len(self.dst_b))

    def __getitem__(self, index):
        a_item = self.dst_a[index % len(self.dst_a)]
        b_item = self.dst_b[index % len(self.dst_b)]
        return (*a_item, *b_item)
