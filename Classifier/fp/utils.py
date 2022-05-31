import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from dgl.data.utils import Subset
from models import MolecularFP
from dataset import USPTODataset, USPTOTestDataset,USPTOvisDataset




def load_dataloader(args,rank):
    if args['mode'] == 'train':
        if rank == "high":
            dataset = USPTODataset(args,"high")
            train_set, val_set= Subset(dataset, dataset.train_ids), Subset(dataset, dataset.val_ids)
            train_loader = DataLoader(dataset=train_set, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_workers'])
            val_loader = DataLoader(dataset=val_set, batch_size=args['batch_size'], num_workers=args['num_workers'])
            return train_loader, val_loader
        if rank == "middle":
            dataset = USPTODataset(args,"middle")
            train_set, val_set= Subset(dataset, dataset.train_ids), Subset(dataset, dataset.val_ids)
            train_loader = DataLoader(dataset=train_set, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_workers'])
            val_loader = DataLoader(dataset=val_set, batch_size=args['batch_size'], num_workers=args['num_workers'])
            return train_loader, val_loader
        if rank == "low":
            dataset = USPTODataset(args, "low")
            train_set, val_set = Subset(dataset, dataset.train_ids), Subset(dataset, dataset.val_ids)
            train_loader = DataLoader(dataset=train_set, batch_size=args['batch_size'], shuffle=True,
                                      num_workers=args['num_workers'])
            val_loader = DataLoader(dataset=val_set, batch_size=args['batch_size'], num_workers=args['num_workers'])
            return train_loader, val_loader
    elif args['mode'] == 'train_2':
        if rank == "high":
            test_set = USPTOTestDataset(args,"high")
            test_loader = DataLoader(dataset=test_set, batch_size=args['batch_size'], num_workers=args['num_workers'])
            return test_loader
        if rank == "middle":
            test_set = USPTOTestDataset(args,"middle")
            test_loader = DataLoader(dataset=test_set, batch_size=args['batch_size'], num_workers=args['num_workers'])
            return test_loader
        if rank == "low":
            test_set = USPTOTestDataset(args,"low")
            test_loader = DataLoader(dataset=test_set, batch_size=args['batch_size'], num_workers=args['num_workers'])
            return test_loader
    elif args['mode'] == 'train_3':
            dataset = USPTODataset(args,rank)
            cluster = dataset.cluster
            train_set, val_set = Subset(dataset, dataset.train_ids), Subset(dataset, dataset.val_ids)

            train_loader = DataLoader(dataset=train_set, batch_size=args['batch_size'], shuffle=True,
                                      num_workers=args['num_workers'])
            val_loader = DataLoader(dataset=val_set, batch_size=args['batch_size'], num_workers=args['num_workers'])
            return train_loader, val_loader,cluster
    elif args['mode'] == 'vis':
            vis_set = USPTOvisDataset(args)
            vis_loader = DataLoader(dataset=vis_set, batch_size=args['batch_size'], num_workers=args['num_workers'])
            return vis_loader
    else:
            test_set = USPTOTestDataset(args)
            test_loader = DataLoader(dataset=test_set, batch_size=args['batch_size'],num_workers=args['num_workers'])
            return test_loader

def load_model(args,learning_rate,weight_decay, schedule_step,drop_out ):
    model = MolecularFP(
        node_in_feats=args["nbit"],
        node_hidden_feats=2048,
        label_n=args["cluster"],
        drop_out = drop_out)
    model = model.to(args['device'])

    loss_criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=schedule_step)
    return model, loss_criterion, optimizer, scheduler



def predict(model, fp):
    return model(fp)