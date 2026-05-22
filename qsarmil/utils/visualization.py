from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol
import numpy as np
from IPython.display import display, HTML


def visualize_conformers_grid(mol, weights, key_conformers, top_n=5,
                              style="stick", n_cols=3, width=250, height=250,
                              show_all=False, sort_by_weight=True):
    
    num_confs = mol.GetNumConformers()
    if num_confs != len(weights):
        raise ValueError("Number of weights must equal number of conformers")

    # top-N predicted indices
    top_indices = set(np.argsort(weights)[-top_n:][::-1])
    key_conformers = set(key_conformers)

    if show_all:
        conf_indices = list(range(num_confs))
    else:
        conf_indices = sorted(key_conformers.union(top_indices))

    # sort conformers by weight if requested
    if sort_by_weight:
        conf_indices = sorted(conf_indices, key=lambda i: weights[i], reverse=True)

    viewers_html = []
    for i in conf_indices:
        conf = mol.GetConformer(int(i) + 1)
        block = Chem.MolToMolBlock(mol, confId=conf.GetId())

        color = "0xAAAAAA"  # default grey
        label = f"Conf {i} (w={weights[i]:.2f})"
        if i in key_conformers:
            color = "0xFF0000"  # red
            label += " [TRUE]"
        elif i in top_indices:
            color = "0x0000FF"  # blue
            label += " [PRED]"

        viewer = py3Dmol.view(width=width, height=height)
        viewer.addModel(block, "sdf")
        viewer.setStyle({style: {"color": color}})
        viewer.zoomTo()

        html = viewer._make_html()
        viewers_html.append(f"<div style='display:inline-block; text-align:center;'>{html}<br>{label}</div>")

    # arrange into grid
    rows = []
    for i in range(0, len(viewers_html), n_cols):
        row_html = "".join(viewers_html[i:i + n_cols])
        rows.append(f"<div style='margin-bottom:20px'>{row_html}</div>")

    # add legend
    legend_html = """
    <div style='margin:10px 0;'>
      <b>Legend:</b> 
      <span style='color:red;'>[TRUE]=Ground truth</span> | 
      <span style='color:blue;'>[PRED]=Top predicted</span> | 
      <span style='color:gray;'>Others</span>
    </div>
    """

    display(HTML(legend_html + "".join(rows)))