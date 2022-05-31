from argparse import ArgumentParser
import torch
import torch.nn as nn
from utils import load_model, load_dataloader, predict
from Train_classifier import calculate_top_k_accuracy, get_top_k_result
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from scipy import interp
from sklearn.metrics import roc_curve, auc,average_precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from itertools import cycle
from sklearn.metrics import precision_recall_curve


def vis(args, model, vis_loader):
    model.eval()
    val_loss = 0
    val_acc = 0
    all_soft_pred = []
    all_pred = []
    all_labels = []
    top_1_acc, top_3_acc, top_5_acc, top_10_acc, nlens = 0,0,0,0,0
    with torch.no_grad():
        for batch_id, batch_data in enumerate(vis_loader):
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
            top_1_acc += calculate_top_k_accuracy(a, y_label, k=1) * lens
            top_3_acc += calculate_top_k_accuracy(a, y_label) * lens
            top_5_acc += calculate_top_k_accuracy(a, y_label, k=5) * lens
            top_10_acc += calculate_top_k_accuracy(a, y_label, k=10) * lens
            _, indices = torch.max(logits, dim=1)

            all_labels += labels.tolist()

            soft = nn.Softmax(dim=1)
            soft_logits = soft(logits).tolist()

            all_soft_pred += soft_logits
            all_pred += indices.tolist()

            correct = torch.sum(indices == labels)
            acc_a = correct.item() * 1.0 / len(labels)

            total_acc = acc_a

            val_acc += total_acc
        lb = LabelBinarizer()
        label_one_hot = lb.fit_transform(all_labels)

        n_classes = label_one_hot.shape[1]
        y_label = label_one_hot
        y_score = np.array(all_soft_pred)
        #ROC
        fpr,tpr,roc_auc = dict(),dict(),dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_label[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        #micro
        fpr["micro"], tpr["micro"], _ = roc_curve(y_label.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # macro
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        #Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        macro_max = 0.1
        if roc_auc["macro"] >=macro_max:
            # Plot all ROC curves
            lw = 0.5
            plt.plot([1, 2, 3])
            plt.subplot(211)
            plt.figure(dpi=300,figsize=(10,10))
            plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                  ''.format(roc_auc["micro"]),
                     color='deeppink', linestyle='-', linewidth=1.5)

            plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
                  color='navy', linestyle='-', linewidth=1.5)

            colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
            for i, color in zip(range(3), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=1.2,
                label='ROC curve of class {0} (area = {1:0.2f})'
                    ''.format(i+1, roc_auc[i]))
            for i, color in zip(range(30,33), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=1.2,
                label='ROC curve of class {0} (area = {1:0.2f})'
                    ''.format(i+1, roc_auc[i]))
            for i, color in zip(range(60,63), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=1.2,
                label='ROC curve of class {0} (area = {1:0.2f})'
                    ''.format(i+1, roc_auc[i]))

            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            ###设置坐标轴的粗细
            ax = plt.gca();  # 获得坐标轴的句柄
            ax.spines['bottom'].set_linewidth(2)  ###设置底部坐标轴的粗细
            ax.spines['left'].set_linewidth(2) ####设置左边坐标轴的粗细
            ax.spines['right'].set_linewidth(2)
            ax.spines['top'].set_linewidth(2)  ###设置右边坐标轴的粗细 ###设置右边坐标轴的粗细
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel('False Positive Rate',fontsize=20)
            plt.ylabel('True Positive Rate',fontsize=20)
            plt.title('Multi-calss ROC',fontsize=20)
            plt.legend(loc="best",fontsize=15)
            plt.savefig("123.png")
    #PR
    precision,recall,average_precision = dict(),dict(),dict()
    n_classes = label_one_hot.shape[1]
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_label[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_label[:, i], y_score[:, i])

    # A "macro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_label.ravel(),
                                                                    y_score.ravel())
    average_precision["micro"] = average_precision_score(y_label, y_score,
                                                         average="micro")
    all_fpr = np.unique(np.concatenate([precision[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, precision[i], recall[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    precision["macro"] = all_fpr
    recall["macro"] = mean_tpr

    if roc_auc["macro"] >= macro_max:
        # Plot all ROC curves
        plt.clf()
        colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])
        plt.figure(dpi=300, figsize=(10, 10))
        _, ax = plt.subplots(figsize=(10, 10))

        f_scores = np.linspace(0.2, 0.8, num=4)
        lines, labels = [], []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
            plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

        display = PrecisionRecallDisplay(
            recall=recall["micro"],
            precision=precision["micro"],
            average_precision=average_precision["micro"],
        )
        display.plot(ax=ax, name="Micro-average PR", color="gold")

        for i, color in zip(range(3), colors):
            display = PrecisionRecallDisplay(
                recall=recall[i],
                precision=precision[i],
                average_precision=average_precision[i],
            )
            display.plot(ax=ax, name=f"PR for class {i+1}", color=color,lw=1.2)
        for i, color in zip(range(30,33), colors):
            display = PrecisionRecallDisplay(
                recall=recall[i],
                precision=precision[i],
                average_precision=average_precision[i],
            )
            display.plot(ax=ax, name=f"PR for class {i+1}", color=color,lw=1.2)


        # add the legend for the iso-f1 curves
        handles, labels = display.ax_.get_legend_handles_labels()
        handles.extend([l])
        labels.extend(["iso-f1 curves"])
        # set the legend and the axes
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        ax.legend(handles=handles, labels=labels, loc=3,fontsize=15)
        plt.xlabel('Recall', fontsize=20)
        plt.ylabel('Precision', fontsize=20)
        ax.set_title("Extension of Precision-Recall curve to multi-class",fontsize=20)
        ax = plt.gca();  # 获得坐标轴的句柄
        ax.spines['bottom'].set_linewidth(2)  ###设置底部坐标轴的粗细
        ax.spines['left'].set_linewidth(2)  ####设置左边坐标轴的粗细
        ax.spines['right'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)  ###设置右边坐标轴的粗细
        plt.savefig("456.png")
    # draw the pipline
    plt.clf()
    plt.figure(dpi=300, figsize=(10, 10))
    cc=pd.read_csv("../../clustering/scripts/data/cluster_train_valid_50.csv")
    plt.style.use('seaborn-white')
    kwargs = dict(histtype='stepfilled', alpha=0.3, bins=10,label="superclass_type")
    plt.hist(list(cc["class"]),**kwargs)
    kwargs2 = dict(histtype='stepfilled', alpha=0.3, bins=77,label="clustering_type")
    plt.hist([a+1 for a in list(cc["fp_FC2_r1_cutoff_0.6"])], **kwargs2)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("The distribution of clustering types and superclasses", fontsize=20)
    plt.legend(loc="best", fontsize=20)
    ax = plt.gca();  # 获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(2)  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2)  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    plt.xlabel('Type number', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.savefig("count.png")
    a=[]



    P_macro, R_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_pred, average='macro')
    P_micro, R_micro, f1_micro, _ = precision_recall_fscore_support(all_labels, all_pred, average='micro')
    print("macro_precision is %.6f, macro_Recall is %.6f, macro_fscore is %.6f" % (P_macro, R_macro, f1_macro))
    print("micro_precision is %.6f, micro_Recall is %.6f, micro_fscore is %.6f" % (P_micro, R_micro, f1_micro))
    # print("macro_auc is %.6f, micro_auc is %.6f " % (roc_auc["macro"],roc_auc["micro"]))
    print(
        "val_loss is %.6f, val_top_1_acc is %.6f, val_top_3_acc is %.6f, val_top_5_acc is %.6f, val_top_10_acc is %.6f " % (
        val_loss / batch_id, top_1_acc / nlens, top_3_acc / nlens, top_5_acc / nlens, top_10_acc / nlens))
    return val_loss / batch_id, top_1_acc / batch_id, top_3_acc / batch_id, top_5_acc / batch_id



def main(learning_rate, weight_decay, schedule_step, drop_out, args, test_loader):
    model_name = 'USPTO_50K_optimizer_original_77_fp_reac.pth'
    args['model_path'] = '../models/' + model_name
    model, _, _, _ = load_model(args, learning_rate, weight_decay, int(schedule_step), drop_out)
    checkpoint = torch.load(args['model_path'])
    model.load_state_dict(checkpoint['net'])
    vis(args, model, test_loader)


if __name__ == '__main__':
    parser = ArgumentParser('LocalRetro testing arguements')
    parser.add_argument('-g', '--gpu', default='cuda:0', help='GPU device to use')
    parser.add_argument('-d', '--dataset', default='USPTO_50K', help='Dataset to use')
    parser.add_argument('-b', '--batch-size', default=128, help='Batch size of dataloader')
    parser.add_argument('-n', '--num-epochs', type=int, default=50, help='Maximum number of epochs for training')
    parser.add_argument('-p', '--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('-cl', '--max-clip', type=int, default=20, help='Maximum number of gradient clip')
    parser.add_argument('-clu', '--cluster', type=int, default=77, help='the cluster num of molecules')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.001, help='Learning rate of optimizer')
    parser.add_argument('-l2', '--weight-decay', type=float, default=1e-4, help='Weight decay of optimizer')
    parser.add_argument('-ss', '--schedule_step', type=int, default=10, help='Step size of learning scheduler')
    parser.add_argument('-nw', '--num-workers', type=int, default=0, help='Number of processes for data loading')
    parser.add_argument('-do', '--drop_out', type=int, default=0.8, help='dropout')
    parser.add_argument('-nb', '--nbit', type=int, default=8192, help='the nbits of fingerprint model')
    args = parser.parse_args().__dict__
    args['mode'] = 'vis'
    args["cluster_name"] = "fp_FC2_r1_cutoff_0.6"
    args['device'] = torch.device(args['gpu']) if torch.cuda.is_available() else torch.device('cpu')
    print('Using device %s' % args['device'])
    args['data_dir'] = '../data/%s' % args['dataset']
    vis_loader = load_dataloader(args,rank=False)
    main(args["learning_rate"], args["weight_decay"], args["schedule_step"], args["drop_out"], args, vis_loader)