from torch.utils.data.dataset import Dataset
import pyreadr
import utils as utils

RAND_TRAIN_INDICES = None
RAND_TEST_INDICES = None


class RDataset(Dataset):
    def __init__(self, mode="train", percentage_in_train=0.85):
        global RAND_TRAIN_INDICES, RAND_TEST_INDICES
        results = pyreadr.read_r('../data/superconductivity_data_train.rda')["data_train"].to_numpy()

        
        self.data = results[:, :results.shape[1]-1]
        self.labels = results[:, results.shape[1]-1]
        print(self.data.shape, self.labels.shape)
        print("labels: ", self.data.shape[1]-1,  self.labels[:10])

        if RAND_TRAIN_INDICES == None or RAND_TEST_INDICES == None:
            print("Create random indices.")
            amount = self.labels.shape[0]
            train_amount = int(percentage_in_train * amount)
            test_amount = amount - train_amount

            RAND_TRAIN_INDICES, RAND_TEST_INDICES = utils.create_random_int_arrays(train_amount, test_amount)
        
        if mode == "train":
            self.indices = RAND_TRAIN_INDICES
        else:
            self.indices = RAND_TEST_INDICES


    def __len__(self,):
        return len(self.indices)
    
    def __getitem__(self, idx):
        index = self.indices[idx]
        return self.data[index], self.labels[index]