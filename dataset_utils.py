class DatasetWrapper:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if hasattr(self.dataset, 'process') and hasattr(self.dataset, '_data'):
            return self.dataset.process(self.dataset._data[idx])
        return self.dataset[idx]

