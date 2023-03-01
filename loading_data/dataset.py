# -*-coding:utf-8-*-
from torch.utils.data import Dataset
from typing import Union
from pathlib import Path
import torch
import numpy as np
from Bio import SeqIO

class Configue(object):
    label_list = ['<cls>', '<pad>', '<eos>', '<unk>', 'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q','N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-', '<null_1>', '<mask>']


class FunDataset(Dataset):
    def __init__(self,dir_name:Union[str,Path]):
        self.data,self.id = self.processingData(dir_name)
        self.len = len(self.data)

    def processingData(self,dir_name):
        all_data = []
        all_id = []
        for record in SeqIO.parse(dir_name,"fasta"):
            all_data.append(str(record.seq))
            all_id.append(str(record.id))
        return all_data,all_id

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index:int):
        return self.data[index],self.id[index]

    @staticmethod
    def collate__fn(batch):
        seq,ids = tuple(zip(*batch))
        max_len = max(len(seq[i]) for i in range(len(seq)))
        batch_size = len(batch)
        if max_len > 1022:
            tokens = torch.empty((batch_size, 1024), dtype=torch.int64)
            tokens.fill_(Configue.label_list.index('<pad>'))
            for i in range(batch_size):
                tokens[i, 0] = Configue.label_list.index("<cls>")
                len_pro = len(seq[i])
                if len_pro > 1022:
                    len_pro = 1022
                seq_get = torch.tensor(
                    [Configue.label_list.index(item) if item in Configue.label_list else Configue.label_list.index(
                        '<unk>') for p, item in enumerate(seq[i])], dtype=torch.int64
                )
                tokens[i, 1:len_pro + 1] = seq_get[:len_pro]
                tokens[i, len_pro + 1] = Configue.label_list.index("<eos>")
        else:
            tokens = torch.empty((batch_size, max_len + 2), dtype=torch.int64)
            tokens.fill_(Configue.label_list.index('<pad>'))
            for i in range(batch_size):
                tokens[i, 0] = Configue.label_list.index("<cls>")
                len_pro = len(seq[i])
                seq_get = torch.tensor(
                    [Configue.label_list.index(item) if item in Configue.label_list else Configue.label_list.index(
                        '<unk>') for p, item in enumerate(seq[i])], dtype=torch.int64
                )
                tokens[i, 1:len_pro + 1] = seq_get
                tokens[i, len_pro + 1] = Configue.label_list.index("<eos>")
        return tokens,ids,seq