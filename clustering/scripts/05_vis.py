import pandas as pd
from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints
)
from tqdm import tqdm
import pickle
import os
import tmap as tm
from faerun import Faerun
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

model, tokenizer = get_default_model_and_tokenizer()

rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)


original_f = pd.read_csv("data/cluster_train_valid_50.csv")
rxn_batch = original_f["rxn_smiles"].tolist()

lf = tm.LSHForest(256, 128)
mh_encoder = tm.Minhash()

if os.path.exists("rxn_fps.pkl"):
    with  open(r"rxn_fps.pkl", 'rb')  as f1:
        fps = pickle.load(f1)
else:
    fps = [rxnfp_generator.convert(rxn) for rxn in tqdm(rxn_batch)]
    pickle.dump(fps, open("rxn_fps"+ ".pkl", "wb"))

mhfps = [mh_encoder.from_weight_array(fp, method="I2CWS") for fp in tqdm(fps)]

labels = []
superclasses = []
class_FC2_r1_6 = []
class_MG2_r0_6 = []
class_FC2_r0_6 = []
class_TT_r0_6 = []
class_FC2_r0_6 =[]

for i, row in tqdm(original_f.iterrows(), total=len(original_f)):
    rxn = row["rxn_smiles"]
    labels.append(str(rxn)+ "__"+str(rxn))
    superclasses.append(int(row["class"]))
    class_FC2_r1_6.append(int(row["fp_FC2_r1_cutoff_0.6"]))
    class_MG2_r0_6.append(int(row["fp_MG2_r0_cutoff_0.6"]))
    class_FC2_r0_6.append(int(row["fp_FC2_r0_cutoff_0.6"]))
    class_TT_r0_6.append(int(row["fp_TT_r0_cutoff_0.6"]))
    class_FC2_r0_6.append(int(row["fp_FC2_r0_cutoff_0.6"]))
labels_groups, groups = Faerun.create_categories(superclasses)
labels_groups = [(label[0], str(label[1])+"-class") for label in labels_groups]

lf.batch_add(mhfps)
lf.index()

# Layout
cfg = tm.LayoutConfiguration()
cfg.k = 50
cfg.kc = 50
cfg.sl_scaling_min = 1.0
cfg.sl_scaling_max = 1.0
cfg.sl_repeats = 1
cfg.sl_extra_scaling_steps = 2
cfg.placer = tm.Placer.Barycenter
cfg.merger = tm.Merger.LocalBiconnected
cfg.merger_factor = 2.0
cfg.merger_adjustment = 0
cfg.fme_iterations = 1000
cfg.sl_scaling_type = tm.ScalingType.RelativeToDesiredLength
cfg.node_size = 1 / 37
cfg.mmm_repeats = 1

# Define colormaps
set1 = plt.get_cmap("Set1").colors
rainbow = plt.get_cmap("tab20c")
colors = rainbow(np.linspace(0, 1, len(set(groups))))[:, :3].tolist()
custom_cm = LinearSegmentedColormap.from_list("my_map", colors, N=len(colors))
colors1 = rainbow(np.linspace(0, 1, len(set(class_FC2_r1_6))))[:, :3].tolist()
colors2 = rainbow(np.linspace(0, 1, len(set(class_MG2_r0_6))))[:, :3].tolist()
colors3 = rainbow(np.linspace(0, 1, len(set(class_FC2_r0_6))))[:, :3].tolist()
colors4 = rainbow(np.linspace(0, 1, len(set(class_TT_r0_6))))[:, :3].tolist()
colors5 = rainbow(np.linspace(0, 1, len(set(class_FC2_r0_6))))[:, :3].tolist()
custom_cm1 = LinearSegmentedColormap.from_list("my_map1", colors1, N=15)
custom_cm2 = LinearSegmentedColormap.from_list("my_map2", colors2, N=15)
custom_cm3 = LinearSegmentedColormap.from_list("my_map3", colors3, N=15)
custom_cm4 = LinearSegmentedColormap.from_list("my_map4", colors4, N=15)
custom_cm5 = LinearSegmentedColormap.from_list("my_map5", colors5, N=15)



# Get tree coordinates
x, y, s, t, _ = tm.layout_from_lsh_forest(lf, config=cfg)

# slow
f = Faerun(clear_color="#ffffff", coords=False, view="front", )

f.add_scatter(
    "ReactionAtlas",
    {
        "x": x, "y": y,
        "c": [
            groups,  # superclasses
            class_FC2_r1_6,
            class_MG2_r0_6,
            class_FC2_r0_6,
            class_TT_r0_6,
            class_FC2_r0_6
        ],
        "labels": labels
    },
    shader="smoothCircle",
    colormap=[
        custom_cm,
        custom_cm1,
        custom_cm2,
        custom_cm3,
        custom_cm4,
        custom_cm5,

    ],
    point_scale=2.0,
    categorical=[
        True,
        True,
        True,
        True,
        True,
        True
    ],
    has_legend=True,
    legend_labels=[
        labels_groups,
        None,
        None,
        None,
        None,
        None
    ],
    selected_labels=["SMILES","SMILES"],
    series_title=[
        "Superclass",
        "class_FC2_r1_6",
        "class_MG2_r0_6",
        "class_FC2_r0_6",
        "class_TT_r0_6",
        "class_FC2_r0_6"
    ],
    max_legend_label=[
        None,
        None,
        None,
        None,
        None,
        None
        #str(round(max(class_AP3_r0_4))),
        #str(round(max(class_MG2_r0_4))),
        #str(round(max(class_FC2_r1_4))),
        #str(round(max(class_TT_r0_4))),
        #str(round(max(class_FC2_r0_5)))
    ],
    min_legend_label=[
        None,
        None,
        None,
        None,
        None,
        None
        #str(round(min(class_AP3_r0_4))),
        #str(round(min(class_MG2_r0_4))),
        #str(round(min(class_FC2_r1_4))),
        #str(round(min(class_TT_r0_4))),
        #str(round(min(class_FC2_r0_5)))
    ],
    title_index=2,
    legend_title="",
)

f.add_tree("reactiontree", {"from": s, "to": t}, point_helper="ReactionAtlas")
plot = f.plot("ft_50k", template="reaction_smiles")

