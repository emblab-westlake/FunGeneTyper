# -*-coding:utf-8-*-
import os
import torch
from torch.utils.data.dataloader import DataLoader
import esm
import argparse
import random
import torch.nn as nn
# from apex import amp
import parameters
# import faiss
import numpy as np
import linecache
from tqdm import tqdm
import collections
from loading_data import RetrieveDatasetTriplet,KernelDatasetTriplet,EmbeddingDatasetTriplet

class Setting(object):
    BATCH_SIZE = 3
    EPOCHS = 500
    LEARNING_RATE = 1e-5
    TRIPLETMARGIN = 1.0

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
devices = "cuda:0" if torch.cuda.is_available() else "cpu"

# load backbone parameters
model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
model.to(devices)

# freeze parameters
for name, param in model.named_parameters():
    if "finetune" in name or "dnn" in name:
        continue
    param.requires_grad = False

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=Setting.LEARNING_RATE)
# model,optimizer = amp.initialize(model,optimizer,opt_level="O2")

TrainData = RetrieveDatasetTriplet("example/ARGs_group_TrainingData/all_train_data.txt")
train_dataset = DataLoader(dataset=TrainData,batch_size=Setting.BATCH_SIZE,shuffle=True,collate_fn=TrainData.collate__fn,drop_last=True,pin_memory=True)

criterion = nn.TripletMarginLoss(margin=Setting.TRIPLETMARGIN)
criterion.to(devices)

Test_big_label_Acc = 0
Test_small_label_Acc = 0
Validation_big_label_Acc = 0
Validation_small_label_Acc = 0
model.train()
for epoch in range(Setting.EPOCHS):
    for i, tData in enumerate(train_dataset):
        seq_1, seq_2, seq_3 = tData
        anchor = seq_1.cuda()
        positive = seq_2.cuda()
        negitive = seq_3.cuda()

        result_1 = model(anchor, repr_layers=[33])
        result_2 = model(positive, repr_layers=[33])
        result_3 = model(negitive, repr_layers=[33])

        # use <cls>
        anchor_out = result_1["representations"][33][:, 0, :].squeeze(1).cuda()
        positive_out = result_2["representations"][33][:, 0, :].squeeze(1).cuda()
        negnative_out = result_3["representations"][33][:, 0, :].squeeze(1).cuda()

        loss = criterion(anchor_out, positive_out, negnative_out)
        print("Epoch: {}. iteraction: {}. loss: {}.".format(epoch, i, loss.item()))

        optimizer.zero_grad()
        loss.backward()
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        optimizer.step()

        if i % 40000 == 0 and i != 0:
            # Vaildation
            model.eval()

            # Convert to embedding
            Kernel_data = KernelDatasetTriplet("examples/ARGs_group_TrainingData/kernel_sequence.txt")
            kernel_data_dataset = DataLoader(dataset=Kernel_data, batch_size=100, shuffle=False,
                                             collate_fn=Kernel_data.collate__fn, drop_last=False, pin_memory=True)

            # Convert to embedding
            vail_data = EmbeddingDatasetTriplet("examples/ARGs_group_TrainingData/validation_last.txt")
            vail_data_dataset = DataLoader(dataset=vail_data, batch_size=100, shuffle=False,
                                           collate_fn=vail_data.collate__fn, drop_last=False, pin_memory=True)

            # Convert to embedding
            test_data = EmbeddingDatasetTriplet("examples/ARGs_group_TrainingData/test_last.txt")
            test_data_dataset = DataLoader(dataset=test_data, batch_size=100, shuffle=False,
                                           collate_fn=test_data.collate__fn, drop_last=False, pin_memory=True)

            out = []
            for m, test_out in tqdm(enumerate(kernel_data_dataset), postfix="Batch: /30 (Batch_size:100)"):
                test_out = test_out.cuda()
                with torch.no_grad():
                    result = model(test_out, repr_layers=[33], return_contacts=False)
                    test_result = result["representations"][33][:, 0, :].cpu().data.numpy()
                out.append(test_result)
            output_kernel = np.concatenate(out, axis=0).astype('float32')

            # validation
            out_vail = []
            vali_label_big_category = []
            vali_label_small_category = []
            for n, vail_out in tqdm(enumerate(vail_data_dataset), postfix="Batch: /118 (Batch_size:100)"):
                reallyVali_out, vali_big_category, vali_little_category = vail_out
                reallyVali_out = reallyVali_out.cuda()
                with torch.no_grad():
                    result = model(reallyVali_out, repr_layers=[33], return_contacts=False)
                    vail_result = result["representations"][33][:, 0, :].cpu().data.numpy()
                out_vail.append(vail_result)
                # concate label
                vali_label_big_category += vali_big_category
                vali_label_small_category += vali_little_category
            output_vail = np.concatenate(out_vail, axis=0).astype('float32')

            # test
            out_test = []
            test_label_big_category = []
            test_label_small_category = []
            for k, reallyTest_out in tqdm(enumerate(test_data_dataset), postfix="Batch: /118 (Batch_size:100)"):
                reallyTest_out, test_big_category, test_little_category = reallyTest_out
                reallyTest_out = reallyTest_out.cuda()
                with torch.no_grad():
                    result = model(reallyTest_out, repr_layers=[33], return_contacts=False)
                    reallyTest_result = result["representations"][33][:, 0, :].cpu().data.numpy()
                out_test.append(reallyTest_result)
                # concate label
                test_label_big_category += test_big_category
                test_label_small_category += test_little_category
            output_test = np.concatenate(out_test, axis=0).astype('float32')

            # faiss
            index = faiss.IndexFlatL2(1280)
            index.add(output_kernel)

            # Validation set for validation
            D, I = index.search(output_vail, k=1)  # Xq is the vector to be retrieved, the returned I is the index list most similar to TopK of each query to be retrieved, and D is its corresponding distance
            predict_label_big = []
            predict_label_small = []
            for num, ids in enumerate(I):
                ids = ids[0]
                label_predict = linecache.getline("examples/ARGs_group_TrainingData/kernel.txt", ids + 1).strip().split("\t")
                predict_label_big.append(label_predict[1])
                predict_label_small.append(label_predict[2])

            with open("examples/ARGs_group_TrainingData/Train/Vali/detailed/" + str(epoch) + "_" + str(i) + ".txt","w") as acc_out:
                for s in range(len(predict_label_big)):
                    acc_out.write(str(vali_label_big_category[s]) + "\t" + str(predict_label_big[s]) + "\t" + str(
                        vali_label_small_category[s]) + "\t" + str(predict_label_small[s]) + "\n")

            vali_big_acc = 0
            vali_small_acc = 0
            for p in range(len(test_label_big_category)):
                if vali_label_big_category[p] == predict_label_big[p]:
                    vali_big_acc += 1
                if vali_label_small_category[p] == predict_label_small[p]:
                    vali_small_acc += 1

            with open("examples/ARGs_group_TrainingData/Train/Vali/detailed/acc.txt", "a+") as acc_out_1:
                acc_out_1.write("Epoch: " + str(epoch) + "\t" + "iteraction: " + str(i) + "\n")
                acc_out_1.write("Big_acc" + str(vali_big_acc / len(predict_label_big)) + "\t" + "small_acc" + str(
                    vali_small_acc / len(predict_label_small)) + "\n")

            if vali_small_acc / len(predict_label_small) > Validation_small_label_Acc:
                Validation_small_label_Acc = vali_small_acc / len(predict_label_small)

                # save model
                ARGs_group = {}
                for name, para in model.named_parameters():
                    if "finetune" in name or "dnn" in name:
                        ARGs_group[name] = para
                torch.save(ARGs_group, "examples/ARGs_group_TrainingData/Train/Model/ARGs_group.pkl")

                # Test
                D, I = index.search(output_test, k=1)

                predict_label_big = []
                predict_label_small = []
                for num, ids in enumerate(I):
                    ids = ids[0]
                    label_predict = linecache.getline("examples/ARGs_group_TrainingData/kernel.txt",ids + 1).strip().split("\t")
                    predict_label_big.append(label_predict[1])
                    predict_label_small.append(label_predict[2])

                with open("examples/ARGs_group_TrainingData/Test/detailed/" + str(epoch) + "_" + str(i) + ".txt", "w") as acc_out:
                    for s in range(len(predict_label_big)):
                        acc_out.write(str(test_label_big_category[s]) + "\t" + str(predict_label_big[s]) + "\t" + str(
                            test_label_small_category[s]) + "\t" + str(predict_label_small[s]) + "\n")

                test_big_acc = 0
                test_small_acc = 0
                for p in range(len(test_label_big_category)):
                    if test_label_big_category[p] == predict_label_big[p]:
                        test_big_acc += 1
                    if test_label_small_category[p] == predict_label_small[p]:
                        test_small_acc += 1

                with open("examples/ARGs_group_TrainingData/Test/acc.txt", "a+") as acc_out_1:
                    acc_out_1.write("Epoch: " + str(epoch) + "\t" + "iteraction: " + str(i) + "\n")
                    acc_out_1.write("Big_acc" + str(test_big_acc / len(predict_label_big)) + "\t" + "small_acc" + str(test_small_acc / len(predict_label_small)) + "\n")
            model.train()