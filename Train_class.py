# -*-coding:utf-8-*-
import numpy as np
import esm
import torch.nn as nn
import torch
from torch.utils.data.dataloader import DataLoader
import os
import csv
import random
import parameters
from loading_data import DatasetTrain
# from apex import amp

class FunGeneTyper(torch.nn.Module):
    def __init__(self, Category):
        super(FunGeneTyper,self).__init__()
        self.modelEsm, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        self.dnn = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1280, Category)
        )

    def forward(self,data):
        result = self.modelEsm(data, repr_layers=[33])
        out_result = result["representations"][33][:, 0, :].squeeze(dim=1)
        out_put = self.dnn(out_result)
        return out_put

def init_seeds(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    init_seeds(2021)

    BATCH_SIZE = 8
    EPOCH = 500

    best_acc = 0
    Vail_Acc = []
    Test_Acc = []

    args = parameters.params_parser()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = FunGeneTyper(args.Train_Category)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)

    model.to(device)
    # freeze backbone, only train adapter layer and dnn
    for name, param in model.named_parameters():
        if "finetune" in name or "dnn" in name:
            continue
        param.requires_grad = False

    '''
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    '''

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

    # To speed up model training, you can choose semi-precision
    # model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

    # loading Data
    Train_data = DatasetTrain("example/ARGs_class_TrainingData/TrainPaddingAll_full.txt")
    train_data_dataset = DataLoader(dataset=Train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=Train_data.collate__fn, drop_last=True, pin_memory=True)

    Vail_data = DatasetTrain("example/ARGs_class_TrainingData/Vail_full.txt")
    vail_data_dataset = DataLoader(dataset=Vail_data, batch_size=100, shuffle=False, collate_fn=Vail_data.collate__fn, drop_last=False, pin_memory=True)

    Test_data = DatasetTrain("example/ARGs_class_TrainingData/Test_full.txt")
    test_data_dataset = DataLoader(dataset=Test_data, batch_size=100, shuffle=False, collate_fn=Test_data.collate__fn,drop_last=False, pin_memory=True)

    for epoch in range(EPOCH):
        for i, item in enumerate(train_data_dataset):
            # train
            model.train()
            content, label = item
            content = content.cuda()
            label = label.cuda()
            last_result = model(content).cuda()
            loss = criterion(last_result, label).cuda()
            print("epoch: {} \t iteration : {} \t Loss: {} \t lr: {}".format(epoch, i, loss.item(),optimizer.param_groups[0]['lr']))

            optimizer.zero_grad()
            loss.backward()
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #    scaled_loss.backward()
            optimizer.step()

            if i % 20000 == 0 and i != 0:
                # eval
                model.eval()
                total = 0
                correct = 0

                predict = {}
                predict_really = {}

                # Each category in the validation set and its corresponding number of sequences
                grand_truth = {19: 90696, 7: 7072, 0: 1182, 12: 655, 6: 134, 16: 237, 2: 464, 5: 697, 10: 519, 11: 410,
                               8: 408, 9: 160, 4: 62, 18: 115, 13: 123, 15: 51, 14: 26, 1: 45, 17: 11, 3: 4}

                for j, vail in enumerate(vail_data_dataset):
                    data_vail, label_vail = vail
                    data_vail = data_vail.to(device)
                    label_vail = label_vail.to(device)

                    with torch.no_grad():
                        last_result_vail = model(data_vail).to(device)

                    # label
                    predicted = torch.argmax(last_result_vail.data, dim=1)

                    predict_label = predicted.cpu().numpy()
                    really_label = label_vail.cpu().numpy()

                    for k in range(len(predict_label)):
                        if predict_label[k] not in predict:
                            predict[predict_label[k]] = 1
                        else:
                            predict[predict_label[k]] += 1

                        if predict_label[k] == really_label[k]:
                            if predict_label[k] not in predict_really:
                                predict_really[predict_label[k]] = 1
                            else:
                                predict_really[predict_label[k]] += 1

                    total += label_vail.size(0)
                    correct += (predicted == label_vail).sum().cpu().item()

                out = ""
                for m in range(len(grand_truth)):
                    if m in predict and m in predict_really:
                        out = out + "Category_" + str(m) + "\t" + "predict_really " + str(predict_really[m]) + "\t" + "predict " + str(predict[m]) + "\t" + "     Precision " + str(predict_really[m] / predict[m]) + "\t" + "  recall" + str(predict_really[m] / grand_truth[m]) + "\n"

                with open("example/ARGs_class_TrainingData/result_vail.txt", "a+", encoding="utf-8") as output:
                    output.write("Epoch_item: {} \t\t iteraction: {} \t\t Correct_num: {} \t\t total: {} \t\t Accuracy on test data: {} \n".format(epoch, i, correct, total, correct / total))
                    output.write(out)

                Vail_Acc.append(correct / total)
                with open("example/ARGs_class_TrainingData/result_vail_ACC.txt", "a+", encoding="utf-8") as output:
                    output.write(str(Vail_Acc) + "\n")

                if correct / total > best_acc:
                    best_acc = correct / total

                    ARGs_class = {}
                    for name, para in model.named_parameters():
                        if "finetune" in name or "dnn" in name:
                                ARGs_class[name] = para
                    torch.save(ARGs_class, "example/ARGs_class_TrainingData/VFs_class.pkl")

                    total_test = 0
                    correct_test = 0

                    predict_test = {}
                    predict_really_test = {}

                    # Each category in the test set and the corresponding number of sequences
                    grand_truth_test = {19: 90697, 7: 7073, 12: 655, 18: 115, 2: 463, 0: 1182, 5: 697, 10: 518, 4: 63,
                                        11: 410, 15: 52, 6: 134, 8: 407, 16: 238, 13: 123, 1: 46, 9: 160, 14: 25, 3: 4,
                                        17: 10}

                    for m, test in enumerate(test_data_dataset):
                        data_test, label_test = test
                        data_test = data_test.to(device)
                        label_test = label_test.to(device)

                        with torch.no_grad():
                            last_result_test = model(data_test).to(device)

                        # label
                        predicted = torch.argmax(last_result_test.data, dim=1)

                        predict_label = predicted.cpu().numpy()
                        really_label = label_test.cpu().numpy()

                        # predict        是预测出来的label
                        # predict_really 是预测正确的label
                        for k in range(len(predict_label)):
                            if predict_label[k] not in predict_test:
                                predict_test[predict_label[k]] = 1
                            else:
                                predict_test[predict_label[k]] += 1

                            if predict_label[k] == really_label[k]:
                                if predict_label[k] not in predict_really_test:
                                    predict_really_test[predict_label[k]] = 1
                                else:
                                    predict_really_test[predict_label[k]] += 1

                        total_test += label_test.size(0)
                        correct_test += (predicted == label_test).sum().cpu().item()

                    out = ""
                    for m in range(len(grand_truth_test)):
                        if m in predict_test and m in predict_really_test:
                            out = out + "Category_" + str(m) + "\t" + "predict_really " + str(predict_really_test[m]) + \
                                  "\t" + "predict " + str(predict_test[m]) + "\t" + "     Precision " + str(
                                predict_really_test[m] / predict_test[m]) + "\t" \
                                  + "  recall" + str(predict_really_test[m] / grand_truth_test[m]) + "\n"

                    with open("example/ARGs_class_TrainingData/result_test.txt", "a+", encoding="utf-8") as output:
                        output.write("Epoch_item: {} \t\t iteraction: {} \t\t Correct_num: {} \t\t total: {} \t\t Accuracy on test data: {} \n".format(
                                epoch, i, correct_test, total_test, correct_test / total_test))
                        output.write(out)

                    Test_Acc.append(correct_test / total_test)
                    with open("example/ARGs_class_TrainingData/result_test_ACC.txt", "a+", encoding="utf-8") as output:
                        output.write(str(Test_Acc) + "\n")
