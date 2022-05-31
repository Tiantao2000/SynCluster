from argparse import ArgumentParser
import torch
import torch.nn as nn
from utils import load_model, load_dataloader, predict
from Train2 import calculate_top_k_accuracy,get_top_k_result
import pandas as pd

def write_edits(args, model, write_loader):
    model.eval()
    test_pred = []
    test_top_10_hard = []
    test_logits = []
    test_top_10_logits = []
    top_1_acc = 0
    top_3_acc = 0
    top_5_acc = 0
    top_10_acc = 0
    lens = 0
    with torch.no_grad():
        for batch_id, data in enumerate(write_loader):
            smiles, fp= data
            num_size = len(smiles)
            fp = fp.to(args["device"])
            ori_logits = predict(model, fp)
            batch_logits = nn.Softmax(dim=1)(ori_logits)
            test_logits = test_logits+batch_logits.tolist()
            new_logits = batch_logits.argmax(dim=1).tolist()
            test_pred += new_logits
            a = batch_logits.cpu().numpy()
            lens += num_size
            #cal_1-3-5-10 topk
            _, top_1_indices = get_top_k_result(a, k=1)
            _, top_3_indices = get_top_k_result(a)
            _, top_5_indices = get_top_k_result(a, k=5)
            top_10_pos, top_10_indices = get_top_k_result(a, k=10)
            test_top_10_hard += top_10_indices.tolist()
            test_top_10_logits += top_10_pos.tolist()
    #print("write_top_1_acc is %.6f, write_top_3_acc is %.6f, write_top_5_acc is %.6f, write_top_10_acc is %.6f " % (top_1_acc/lens,top_3_acc/lens,top_5_acc/lens,top_10_acc/lens))
    sourcefile = pd.read_csv("../../clustering/scripts/data/cluster_test_50k.csv")
    sourcefile["top_1_pred_labels"] = [a for a in test_pred]
    sourcefile["top_10_pred_labels"] = [[b for b in a] for a in test_top_10_hard]
    sourcefile["top_10_pred_logits"] = test_top_10_logits
    sourcefile["logits"] = test_logits
    sourcefile.to_csv("out/cluster_test_small.csv")


def main(learning_rate, weight_decay, schedule_step, drop_out, args, test_loader):
    model_name = 'USPTO_50k_optimizer_original_77_fp.pth'
    args['model_path'] = '../models/' + model_name
    model, _, _, _ = load_model(args, learning_rate, weight_decay, int(schedule_step), drop_out)
    checkpoint = torch.load(args['model_path'])
    model.load_state_dict(checkpoint['net'])
    write_edits(args, model, test_loader)
    

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
    args['mode'] = 'test'
    args['device'] = torch.device(args['gpu']) if torch.cuda.is_available() else torch.device('cpu')
    print('Using device %s' % args['device'])
    args['data_dir'] = '../data/%s' % args['dataset']
    test_loader = load_dataloader(args,rank=False)
    args['result_path'] = '../outputs/fp/%s_out_test.csv' %args["cluster"]
    main(args["learning_rate"], args["weight_decay"], args["schedule_step"], args["drop_out"], args, test_loader)