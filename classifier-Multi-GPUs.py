# -*-coding:utf-8-*-
import argparse
import esm
import torch.nn as nn
import torch
from torch.utils.data.dataloader import DataLoader
import os
import parameters
from loading_data import FunDataset
import collections
import faiss
import numpy as np
import linecache
from tqdm import tqdm
from Config import Test_dict
import sys



class FunGeneTyper(torch.nn.Module):
    def __init__(self,args,Category):
        super(FunGeneTyper,self).__init__()
        esm1b_alphabet = esm.data.Alphabet.from_architecture(args.arch)
        self.modelEsm = esm.model.ProteinBertModel(args, esm1b_alphabet)
        self.dnn = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1280, Category)
        )

    def forward(self,data):
        result = self.modelEsm(data, repr_layers=[33])
        out_result = result["representations"][33][:, 0, :].squeeze(dim=1)
        out_put = self.dnn(out_result)
        return out_put

if __name__ == '__main__':
    # 1) Multi-card initialization
    torch.distributed.init_process_group(backend='nccl')

    # 2) local_rank
    args = parameters.params_parser()

    # 3) cudaargs.model
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    world_size = torch.distributed.get_world_size()
    model = FunGeneTyper(args,Test_dict[args.adapter]["Category"])
    model.to(device)

    # load Backbone parameters
    model.load_state_dict(torch.load(sys.path[0]+"/"+"Pretrained_model/backbone.pkl"),strict=False) #TODO：修改为获取脚本的相对路径
    # load adapter and dnn parameters of class model
    model.load_state_dict(torch.load(sys.path[0] + "/" + Test_dict[args.adapter]["class_model"]), strict=False) #TODO：修改为获取路径
    model.eval()

    # dataset
    Test_data = FunDataset(args.input)
    test_sample = torch.utils.data.distributed.DistributedSampler(Test_data)
    test_data_dataset = DataLoader(dataset=Test_data, batch_size=args.batch_size, shuffle=False,
                                   collate_fn=Test_data.collate__fn,
                                   drop_last=False,sampler=test_sample)  # , pin_memory=True)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],output_device=args.local_rank) #, find_unused_parameters=True)
    model.to(device)

    output_fasta = args.output + "_output.fasta"
    with open(args.output + ".txt", "a+", encoding="utf-8") as answer:
        answer.write("id \t class_category \n")
        
    for m, test in tqdm(enumerate(test_data_dataset),postfix="annotation proteins (class level)..."):
        test_seq,ids,seq = test
        test_seq = test_seq.to(device)
        with torch.no_grad():
            last_result_test = model(test_seq).to(device)

        predicted = torch.argmax(last_result_test.data, dim=1)
        predict_label = predicted.cpu().numpy()

        # multi-GPU's results
        all_label = [int for _ in range(world_size)]
        torch.distributed.all_gather_object(all_label, predict_label)
        all_ids = ["" for _ in range(world_size)]
        torch.distributed.all_gather_object(all_ids, ids)
        all_seq = ["" for _ in range(world_size)]
        torch.distributed.all_gather_object(all_seq, seq)

        if args.local_rank == 0:
            with open(output_fasta, "a+") as writeseq:
                with open(args.output + ".txt", "a+", encoding="utf-8") as answer:
                    for p in range(len(all_label)):
                        for k in range(len(all_label[p])):
                            answer.write(all_ids[p][k] + "\t" + Test_dict[args.adapter]["label_list"][all_label[p][k]] + "\n")
                            if all_label[p][k] != Test_dict[args.adapter]["Category"] - 1:
                                writeseq.write(
                                    ">" + all_ids[p][k] + "&&&" + Test_dict[args.adapter]["label_list"][all_label[p][k]] + "\n")
                                writeseq.write(all_seq[p][k] + "\n")
    if args.group:
        output_group = args.output + "_output_group.txt"
        esm1b_alphabet = esm.data.Alphabet.from_architecture(args.arch)
        model2 = esm.model.ProteinBertModel(args, esm1b_alphabet)
        model2.to(device)

        # load backbone parameters
        state_dict_backbone = {".".join(k.split(".")[1:]): v for k, v in
                               torch.load(sys.path[0]+"/"+"Pretrained_model/backbone.pkl").items() if "modelEsm" in k}
        model2.load_state_dict(collections.OrderedDict(state_dict_backbone), strict=False)

        # load adapter and dnn parameters of group model
        model2.load_state_dict(torch.load(sys.path[0] + "/" + Test_dict[args.adapter]["group_model"]), strict=False)
        model2.eval()

        # dataset
        Test_data_group = FunDataset(output_fasta)
        test_sample_group = torch.utils.data.distributed.DistributedSampler(Test_data_group)
        test_data_dataset_group = DataLoader(dataset=Test_data_group, batch_size=args.batch_size, shuffle=False,
                                             collate_fn=Test_data_group.collate__fn,
                                             drop_last=False,sampler=test_sample_group)  # , pin_memory=True)

        model2 = torch.nn.parallel.DistributedDataParallel(model2, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)  # , find_unused_parameters=True)
        model2.to(device)

        # faiss
        data = []
        with open(sys.path[0] + "/" + Test_dict[args.adapter]["CoreEmbedding"], 'rb') as f:
            for line in f:
                temp = line.split()
                data.append(temp)
        dataArray = np.array(data).astype('float32')
        # print("All Data Number: ", len(dataArray))

        # build index
        index = faiss.IndexFlatL2(1280)
        index.add(dataArray)

        for k, test in tqdm(enumerate(test_data_dataset_group), postfix="annotation proteins (group level)..."):
            test_seq, ids, seq = test
            test_seq = test_seq.to(device)
            with torch.no_grad():
                result = model2(test_seq, repr_layers=[33], return_contacts=False)
                Rep = result["representations"][33][:, 0, :].squeeze(dim=1).cpu()
            dataTest = np.array(Rep).astype('float32')
            D, I = index.search(dataTest, k=1)

            # multi-GPU's results
            all_group = [int for _ in range(world_size)]
            torch.distributed.all_gather_object(all_group, I)
            all_ids = ["" for _ in range(world_size)]
            torch.distributed.all_gather_object(all_ids, ids)

            if args.local_rank == 0:
                with open(output_group, "a+", encoding="utf-8") as write:
                    for m in range(len(all_group)):
                        for n in range(len(all_group[m])):
                            id = all_group[m][n][0]
                            label_predict = linecache.getline(Test_dict[args.adapter]["Core"], id + 1).strip().split("\t")
                            write.write(all_ids[m][n].split("&&&")[0] + "\t" + str(label_predict[1:]) + "\n")
