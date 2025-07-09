from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, BRICS
import numpy as np
from typing import List, Tuple
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
import matplotlib.pyplot as plt

# Supported RDKit molecular property functions
PROPERTY_FUNCTIONS = {
    "LogP": Descriptors.MolLogP,
    "MolWt": Descriptors.MolWt,
    "TPSA": rdMolDescriptors.CalcTPSA,
    "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "MolMR": Descriptors.MolMR,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "RingCount": Descriptors.RingCount,
    "FractionCSP3": Descriptors.FractionCSP3,
}

def create_fragment_bags(
    mols: List[Chem.Mol],
    bag_size: int = 5,
    property_name: str = "LogP",
    random_state: int = 42
) -> Tuple[List[List[Chem.Mol]], List[float], List[List[float]]]:
    """
    Create bags of BRICS fragments and compute bag-level label as the sum of a chosen molecular property.

    Parameters:
    - mols: list of RDKit Mol objects
    - fragments_per_bag: number of fragments to sample per molecule
    - property_name: RDKit property to calculate per fragment (e.g., "LogP", "MolWt")
    - random_state: for reproducible fragment sampling

    Returns:
    - bags: list of bags (each a list of fragment Mol objects)
    - labels: list of total property per bag
    - fragment_props: list of property values per fragment per bag
    """

    if property_name not in PROPERTY_FUNCTIONS:
        raise ValueError(f"Unsupported property: {property_name}")

    get_property = PROPERTY_FUNCTIONS[property_name]
    rng = np.random.RandomState(random_state)

    bags = []
    labels = []
    fragment_props = []

    for mol in mols:
        if mol is None:
            continue

        # Generate BRICS fragments
        frag_smiles_set = BRICS.BRICSDecompose(mol)
        frags = [Chem.MolFromSmiles(smi) for smi in frag_smiles_set if smi]
        frags = [f for f in frags if f is not None]

        if len(frags) < bag_size:
            continue  # skip molecules with too few fragments

        # Randomly sample fragments
        sampled_frags = rng.choice(frags, size=bag_size, replace=False).tolist()

        # Compute property per fragment
        props = [get_property(f) for f in sampled_frags]
        total = float(np.sum(props))

        bags.append(sampled_frags)
        labels.append(total)
        fragment_props.append(props)

    return bags, labels, fragment_props


def display_fragments_with_weights(fragments, props, pred_weights, sort=True, max_fragments=16, title=None):

    props = np.array(props)
    pred_weights = np.array(pred_weights)

    if sort:
        sorted_idx = np.argsort(props)[::-1]
        fragments = [fragments[i] for i in sorted_idx]
        props = props[sorted_idx]
        pred_weights = pred_weights[sorted_idx]

    fragments = fragments[:max_fragments]
    props = props[:max_fragments]
    pred_weights = pred_weights[:max_fragments]

    cols = 4
    rows = (len(fragments) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for ax in axes[len(fragments):]:
        ax.axis('off')

    for i, (frag, prop, weight) in enumerate(zip(fragments, props, pred_weights)):
        ax = axes[i]
        img = Draw.MolToImage(frag, size=(150, 150))
        ax.imshow(img)
        ax.set_title(f"True prop: {prop:.3f}\nWeight: {weight:.2f}", fontsize=10)
        ax.axis('off')

    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
    plt.show()