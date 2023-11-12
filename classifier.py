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


class FunGeneTyper(torch.nn.Module):
    def __init__(self, args, Category):
        super(FunGeneTyper, self).__init__()
        esm1b_alphabet = esm.data.Alphabet.from_architecture(args.arch)
        self.modelEsm = esm.model.ProteinBertModel(args, esm1b_alphabet)
        self.dnn = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1280, Category)
        )

    def forward(self, data):
        result = self.modelEsm(data, repr_layers=[33])
        out_result = result["representations"][33][:, 0, :].squeeze(dim=1)
        out_put = self.dnn(out_result)
        return out_put


if __name__ == '__main__':
    args = parameters.params_parser()
    if not args.nogpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"

    model = FunGeneTyper(args, Test_dict[args.adapter]["Category"])
    model.to(device)

    # load Backbone parameters
    model.load_state_dict(
        torch.load("Pretrained_model/backbone.pkl"), strict=False) if not args.nogpu else model.load_state_dict(
        torch.load("Pretrained_model/backbone.pkl", map_location="cpu"), strict=False)

    # load adapter and dnn parameters of class model
    model.load_state_dict(
        torch.load(Test_dict[args.adapter]["class_model"]), strict=False) if not args.nogpu else model.load_state_dict(
        torch.load(Test_dict[args.adapter]["class_model"], map_location="cpu"), strict=False)
    model.eval()

    # dataset
    Test_data = FunDataset(args.input)
    test_data_dataset = DataLoader(dataset=Test_data, batch_size=args.batch_size, shuffle=False,
                                   collate_fn=Test_data.collate__fn,
                                   drop_last=False)  # , pin_memory=True)

    output_fasta = args.output + "_output.fasta"
    with open(args.output+".txt", "a+", encoding="utf-8") as answer:
        answer.write("id \t class_category \n")
        
    for m, test in tqdm(enumerate(test_data_dataset), postfix="annotation proteins (class level)..."):
        test_seq, ids, seq = test
        test_seq = test_seq.to(device)
        with torch.no_grad():
            last_result_test = model(test_seq).to(device)

        # label
        predicted = torch.argmax(last_result_test.data, dim=1)
        predict_label = predicted.cpu().numpy()

        with open(output_fasta, "a+") as writeseq:
            with open(args.output+".txt", "a+", encoding="utf-8") as answer:
                
                for p in range(len(predict_label)):
                    answer.write(ids[p] + "\t" + Test_dict[args.adapter]["label_list"][predict_label[p]] + "\n")
                    if predict_label[p] != Test_dict[args.adapter]["Category"] - 1:
                        writeseq.write(
                            ">" + ids[p] + "&&&" + Test_dict[args.adapter]["label_list"][predict_label[p]] + "\n")
                        writeseq.write(seq[p] + "\n")
    print("Class level prediction Over!")

    # group level classification
    if args.group:
        esm1b_alphabet = esm.data.Alphabet.from_architecture(args.arch)
        model2 = esm.model.ProteinBertModel(args, esm1b_alphabet)
        model2.to(device)

        # load backbone parameters
        if not args.nogpu:
            state_dict_backbone = {".".join(k.split(".")[1:]): v for k, v in
                                torch.load("Pretrained_model/backbone.pkl").items() if "modelEsm" in k}
            model2.load_state_dict(collections.OrderedDict(state_dict_backbone), strict=False)
        else:
            state_dict_backbone = {".".join(k.split(".")[1:]): v for k, v in
                                torch.load("Pretrained_model/backbone.pkl", map_location="cpu").items() if "modelEsm" in k}
            model2.load_state_dict(collections.OrderedDict(state_dict_backbone), strict=False)        

        # load adapter and dnn parameters of group model
        #model2.load_state_dict(torch.load(Test_dict[args.adapter]["group_model"]), strict=False)
        model2.load_state_dict(
        torch.load(Test_dict[args.adapter]["group_model"]), strict=False) if not args.nogpu else model2.load_state_dict(
        torch.load(Test_dict[args.adapter]["group_model"], map_location="cpu"), strict=False)

        model2.eval()

        # dataset
        Test_data_group = FunDataset(output_fasta)
        test_data_dataset_group = DataLoader(dataset=Test_data_group, batch_size=args.batch_size, shuffle=False,
                                             collate_fn=Test_data_group.collate__fn,
                                             drop_last=False)  # , pin_memory=True)
        # faiss
        data = []
        with open(Test_dict[args.adapter]["CoreEmbedding"], 'rb') as f:
            for line in f:
                temp = line.split()
                data.append(temp)
        dataArray = np.array(data).astype('float32')
        # print("All Data Number: ", len(dataArray))

        # build index
        index = faiss.IndexFlatL2(1280)
        index.add(dataArray)

        output_group = args.output + "output_group.txt"
        with open(output_group, "a+", encoding="utf-8") as write:
            write.write("id \t class_category \t group_category \t group_category_detailed \n")
            for k, test in tqdm(enumerate(test_data_dataset_group), postfix="annotation proteins (group level)..."):
                test_seq, ids, seq = test
                test_seq = test_seq.to(device)
                with torch.no_grad():
                    result = model2(test_seq, repr_layers=[33], return_contacts=False)
                    Rep = result["representations"][33][:, 0, :].squeeze(dim=1).cpu()
                dataTest = np.array(Rep).astype('float32')
                D, I = index.search(dataTest, k=1)
                for num, id_out in enumerate(I):
                    id = id_out[0]
                    label_predict = linecache.getline(Test_dict[args.adapter]["Core"], id + 1).strip().split("\t")
                    write.write(ids[num].split("&&&")[0] + "\t" + str(label_predict[1:]) + "\n")
        print("Group level prediction Over!")
