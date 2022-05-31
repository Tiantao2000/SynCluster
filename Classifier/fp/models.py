import torch.nn as nn


class MolecularFP(nn.Module):
    def __init__(self,
                 node_in_feats,
                 node_hidden_feats,
                 label_n,
                 drop_out):
        super(MolecularFP, self).__init__()

        self.label_editor =  nn.Sequential(
                            nn.Linear(node_in_feats, node_hidden_feats),
                            nn.ReLU(),
                            nn.Dropout(drop_out),
                            nn.Linear(node_hidden_feats, label_n)
                            )

    def forward(self,morgan_fp):  #dim num_atom*320?   # num_edges*13
        label_feats3 = morgan_fp.float()
        label_out = self.label_editor(label_feats3)  #dim atom_atom [num_atom,n_atom_template]
        return label_out


