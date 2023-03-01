# -*-coding:utf-8-*-
from torch.utils.data import Dataset
from typing import Union
from pathlib import Path
import numpy as np
import torch

class Configue(object):
    label_list = ['<cls>', '<pad>', '<eos>', '<unk>', 'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q',
                  'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-', '<null_1>', '<mask>']

class RetrieveDatasetTriplet(Dataset):
    def __init__(self,
                 dir_name: Union[str, Path]):
        if not dir_name.endswith(".txt"):
            raise ValueError("Error input")

        self.alldata = np.loadtxt(dir_name, delimiter="\t", dtype=list)
        self.data_origin = self.alldata[:, 0]
        self.data_positive = self.alldata[:, 1]
        self.data_negative = self.alldata[:, 2]

        self.len = self.alldata.shape[0]

    def __len__(self) ->int:
        return self.len

    def __getitem__(self, index: int):
        return self.data_origin[index], self.data_positive[index], self.data_negative[index]

    @staticmethod
    def collate__fn(batch):
        origin, positive, negative = tuple(zip(*batch))
        max_seq_origin = max(len(seq_x) for seq_x in origin)
        max_seq_positive = max(len(seq_y) for seq_y in positive)
        max_seq_negative = max(len(seq_z) for seq_z in negative)
        batch_size = len(batch)

        # using <cls> and <eos>
        tokens_origin = torch.empty((batch_size, max_seq_origin + 2), dtype=torch.int64)
        tokens_positive = torch.empty((batch_size, max_seq_positive + 2), dtype=torch.int64)
        tokens_negative = torch.empty((batch_size, max_seq_negative + 2), dtype=torch.int64)

        tokens_origin.fill_(Configue.label_list.index('<pad>'))
        tokens_positive.fill_(Configue.label_list.index('<pad>'))
        tokens_negative.fill_(Configue.label_list.index('<pad>'))

        for i in range(batch_size):
            len_pro = len(origin[i])
            tokens_origin[i, 0] = Configue.label_list.index("<cls>")
            seq_get = torch.tensor(
                [Configue.label_list.index(item) if item in Configue.label_list else Configue.label_list.index('<unk>')
                for p, item in enumerate(origin[i])], dtype=torch.int64)
            tokens_origin[i, 1:len_pro + 1] = seq_get
            tokens_origin[i, len_pro + 1] = Configue.label_list.index("<eos>")

            len_pro_1 = len(positive[i])
            tokens_positive[i, 0] = Configue.label_list.index("<cls>")
            seq_get_1 = torch.tensor(
                [Configue.label_list.index(item) if item in Configue.label_list else Configue.label_list.index('<unk>')
                 for p, item in enumerate(positive[i])], dtype=torch.int64)
            tokens_positive[i, 1:len_pro_1 + 1] = seq_get_1
            tokens_positive[i, len_pro_1 + 1] = Configue.label_list.index("<eos>")

            len_pro_2 = len(negative[i])
            tokens_negative[i, 0] = Configue.label_list.index("<cls>")
            seq_get_2 = torch.tensor(
                [Configue.label_list.index(item) if item in Configue.label_list else Configue.label_list.index('<unk>')
                 for p, item in enumerate(negative[i])], dtype=torch.int64)
            tokens_negative[i, 1:len_pro_2 + 1] = seq_get_2
            tokens_negative[i, len_pro_2 + 1] = Configue.label_list.index("<eos>")

        return tokens_origin, tokens_positive, tokens_negative

# core database
class KernelDatasetTriplet(Dataset):
    def __init__(self,dir_name:Union[str,Path]):
        self.all_data = np.loadtxt(dir_name,dtype=str)
        self.data = self.all_data[:,1]

        self.len = self.all_data.shape[0]

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index:int):
        return self.data[index]

    @staticmethod
    # The maximum sequence length cannot exceed 1022
    def collate__fn(batch):
        max_len_data = max(len(batch[i]) for i in range(len(batch)))
        batch_size = len(batch)
        tokens = torch.empty(
            (
                batch_size,
                max_len_data + 2  # 填充<cls>和<eos>
            ),
            dtype=torch.int64,
        )
        tokens.fill_(Configue.label_list.index('<pad>'))

        for i in range(batch_size):
            tokens[i, 0] = Configue.label_list.index("<cls>")
            seq_get = torch.tensor(
                [Configue.label_list.index(item) if item in Configue.label_list else Configue.label_list.index(
                    '<unk>') for p, item in enumerate(batch[i])], dtype=torch.int64
            )
            tokens[i, 1:len(batch[i]) + 1] = seq_get
            tokens[i, len(batch[i]) + 1] = Configue.label_list.index("<eos>")
        return tokens

# Validation and Test Data
class EmbeddingDatasetTriplet(Dataset):
    def __init__(self, dir_name:Union[str,Path]):
        self.all_data = np.loadtxt(dir_name, delimiter="\t", dtype=list)
        # id \t big_category \t little_category \t little_category_2 \t sequence \n
        self.data = self.all_data[:,4]
        self.big_category = self.all_data[:,1]
        self.small_category = self.all_data[:,2]

        self.len = self.all_data.shape[0]

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index:int):
        return self.data[index],self.big_category[index],self.small_category[index]

    @staticmethod
    # The maximum sequence length cannot exceed 1022
    def collate__fn(batch):
        data,big_category,small_category = tuple(zip(*batch))
        max_len_data = max(len(data[i]) for i in range(len(data)))
        batch_size = len(batch)
        tokens = torch.empty(
            (
                batch_size,
                max_len_data + 2  # <cls> and <eos>
            ),
            dtype=torch.int64,
        )
        tokens.fill_(Configue.label_list.index('<pad>'))

        for i in range(batch_size):
            tokens[i, 0] = Configue.label_list.index("<cls>")
            seq_get = torch.tensor(
                [Configue.label_list.index(item) if item in Configue.label_list else Configue.label_list.index(
                    '<unk>') for p, item in enumerate(data[i])], dtype=torch.int64
            )
            tokens[i, 1:len(data[i]) + 1] = seq_get
            tokens[i, len(data[i]) + 1] = Configue.label_list.index("<eos>")
        return tokens,big_category,small_category