from argparse import ArgumentParser

import torch
import torch.nn as nn

from utils import load_model, load_dataloader, predict
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import warnings
from collections import defaultdict
from pycm import *


torch.backends.cudnn.enabled = False
warnings.filterwarnings('ignore')

def get_top_k_result(logits, k=3, sorted=True):
    indices = np.argsort(logits, axis=-1)[:, -k:]
    if sorted:
        tmp = []
        for item in indices:
            tmp.append(item[::-1])
        indices = np.array(tmp)
    values = []
    for idx, item in zip(indices, logits):
        p = item.reshape(1, -1)[:, idx].reshape(-1)
        values.append(p)
    values = np.array(values)
    return values, indices

def calculate_top_k_accuracy(logits, targets, k=3):
    values, indices = get_top_k_result(logits, k=k, sorted=False)
    y = np.reshape(targets, [-1, 1])
    correct = (y == indices) * 1.
    top_k_accuracy = np.mean(correct) * k
    return top_k_accuracy


def run_a_train_epoch(args, model, data_loader, loss_criterion, optimizer,epoch):
    model.train()
    train_loss = 0
    train_acc = 0
    for batch_id, batch_data in enumerate(data_loader):
        smiles, fp, labels = batch_data
        if len(smiles) ==1:
            continue
        fp = fp.to(args["device"])
        labels = labels[0].to(args["device"])
        logits= predict(model,fp)
        _, indices = torch.max(logits, dim=1)
        #FL = FocalLoss()
        #loss_fo = FL(logits, labels, args, epoch)

        correct = torch.sum(indices == labels)

        acc_a = correct.item() * 1.0 / len(labels)

        loss_a = loss_criterion(logits, labels).mean()


        total_loss = loss_a
        total_acc = acc_a
        train_loss += total_loss.item()
        train_acc += total_acc
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args['max_clip'])
        optimizer.step()
       # if batch_id % args['print_every'] == 0:
           # print('\nbatch %d/%d, loss %.4f, acc %.4f' % (batch_id + 1, len(data_loader), total_loss.item(), acc_a), end='', flush=True)
    #print('\ntraining loss: %.4f, training acc: %.4f' % (train_loss/batch_id, train_acc/batch_id))
    return train_acc/batch_id, train_loss/batch_id

def run_an_eval_epoch(args, model, data_loader, loss_criterion,fig):
    model.eval()
    val_loss = 0
    val_acc = 0
    all_soft_pred = []
    all_pred =[]
    all_labels =[]
    top_1_acc = 0
    top_3_acc = 0
    top_5_acc = 0
    top_10_acc = 0
    nlens=0
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, fp, labels = batch_data
            fp = fp.to(args["device"])
            labels = labels[0].to(args["device"])
            logits = predict(model, fp)
            _, indices = torch.max(logits, dim=1)
            y_label = labels.cpu().numpy()
            lens = len(smiles)
            nlens += lens
            # top-3-5-10acc
            a = logits.cpu().numpy()
            _, top_1_indices = get_top_k_result(a, k=1)
            _, top_3_indices = get_top_k_result(a)
            _, top_5_indices = get_top_k_result(a, k=5)
            _, top_10_indices = get_top_k_result(a, k=10)
            top_1_acc+=calculate_top_k_accuracy(a, y_label, k=1)*lens
            top_3_acc+=calculate_top_k_accuracy(a, y_label)*lens
            top_5_acc+=calculate_top_k_accuracy(a, y_label,k=5)*lens
            top_10_acc+=calculate_top_k_accuracy(a, y_label,k=10)*lens

            _, indices = torch.max(logits, dim=1)

            all_labels += labels.tolist()

            soft = nn.Softmax(dim=1)
            soft_logits = soft(logits).tolist()

            all_soft_pred += soft_logits
            all_pred += indices.tolist()

            correct = torch.sum(indices == labels)
            acc_a = correct.item() * 1.0 / len(labels)
            loss_a = loss_criterion(logits, labels).mean()

            total_acc = acc_a

            total_loss = loss_a
            val_acc += total_acc
            val_loss += total_loss.item()
    cm = ConfusionMatrix(actual_vector=all_labels, predict_vector=all_pred)
    P_macro,R_macro,f1_macro,_ = precision_recall_fscore_support(all_labels, all_pred, average='macro')
    P_micro,R_micro,f1_micro,_ = precision_recall_fscore_support(all_labels, all_pred, average='micro')
    print("macro_precision is %.6f, macro_Recall is %.6f, macro_fscore is %.6f" % (P_macro,R_macro,f1_macro))
    print("micro_precision is %.6f, micro_Recall is %.6f, micro_fscore is %.6f" % (P_micro,R_micro,f1_micro))
    #print("macro_auc is %.6f, micro_auc is %.6f " % (roc_auc["macro"],roc_auc["micro"]))
    print("val_loss is %.6f, val_top_1_acc is %.6f, val_top_3_acc is %.6f, val_top_5_acc is %.6f, val_top_10_acc is %.6f " % (val_loss/batch_id, top_1_acc/nlens,top_3_acc/nlens,top_5_acc/nlens,top_10_acc/nlens))
    print("kappa is %.6f"%(cm.overall_stat['Kappa']))
    print("mcc is %.6f"%(cm.overall_stat['Overall MCC']))
    return val_loss/batch_id, top_1_acc/nlens,top_3_acc/nlens,top_5_acc/nlens




def main(learning_rate, weight_decay, schedule_step, drop_out, args, train_loader, val_loader):
    history = defaultdict(list)
    model_name = '%s_optimizer_original_%s_fp.pth' % (args['dataset'],args["cluster_name"])
    args['model_path'] = '../models/' + model_name
    model, loss_criterion, optimizer, scheduler =  load_model(args, learning_rate,weight_decay, int(schedule_step),drop_out)
    val_max_top1 = 0.60
    for epoch in range(50):
        #print("epoch:"+str(epoch))
        train_acc,train_loss = run_a_train_epoch(args, model, train_loader, loss_criterion, optimizer,epoch)
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        val_loss,val_top1,val_top3,val_top5 = run_an_eval_epoch(args,model, val_loader, loss_criterion,"fig_461")
        history['val_top1_acc'].append(val_top1)
        history['val_top3_acc'].append(val_top3)
        history['val_top5_acc'].append(val_top5)
        history['val_loss'].append(val_loss)

        if val_top1 > val_max_top1:
            state = {"net":model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
            dir = args['model_path']
            torch.save(state,dir)
            val_max_top1 = val_top1
    print(args["cluster_name"]+"max_top_1"+"is"+str(val_top1))
    #plot_training_history(history,args)
    return val_max_top1



if __name__ == '__main__':
    parser = ArgumentParser('SynCluster training arguements')
    parser.add_argument('-g', '--gpu', default='cuda:0', help='GPU device to use')
    parser.add_argument('-d', '--dataset', default='USPTO_50k', help='Dataset to use')
    parser.add_argument('-b', '--batch-size', default=128, help='Batch size of dataloader')
    parser.add_argument('-n', '--num-epochs', type=int, default=50, help='Maximum number of epochs for training')
    parser.add_argument('-p', '--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('-cl', '--max-clip', type=int, default=20, help='Maximum number of gradient clip')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.001, help='Learning rate of optimizer')
    parser.add_argument('-l2', '--weight-decay', type=float, default=1e-4, help='Weight decay of optimizer')
    parser.add_argument('-ss', '--schedule_step', type=int, default=10, help='Step size of learning scheduler')
    parser.add_argument('-nw', '--num-workers', type=int, default=0, help='Number of processes for data loading')
    parser.add_argument('-do', '--drop_out', type=int, default=0.8, help='dropout')
    parser.add_argument('-pe', '--print-every', type=int, default=100, help='Print the training progress every X mini-batches')
    parser.add_argument('-nb', '--nbit', type=int, default=8192, help='the bit of fingerprint')
    args = parser.parse_args().__dict__
    args['mode'] = 'train_3'
    args['device'] = torch.device(args['gpu']) if torch.cuda.is_available() else torch.device('cpu')
    print ('Using device %s' % args['device'])
    args['data_dir'] = '../data/%s' % args['dataset']
    rank = False
    results = []
    #modes = ["AP3","MG2","TT","FC2"]
    #template_model = ["r0","r1","r2","r3","default"]
    #cut_offs = [0.5,0.6,0.7]
    modes = ["FC2"]
    template_model = ["r1"]
    cut_offs = [0.6]
    for mode in modes:
        for template in template_model:
            for cut_off in cut_offs:
                args["cluster_name"] = "fp_"+mode+"_"+template+"_cutoff_"+str(cut_off)
                #args["cluster_name"] = "big_cluster"
                print("use fp is" + args["cluster_name"])
                train_loader, val_loader,cluster = load_dataloader(args,rank)
                args["cluster"] = cluster
                print("cluster is "+str(args["cluster"]))
                ls = main(args["learning_rate"], args["weight_decay"], args["schedule_step"], args["drop_out"], args, train_loader, val_loader)
