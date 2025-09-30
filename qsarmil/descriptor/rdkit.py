import numpy as np
from rdkit.Chem import Descriptors3D

def validate_desc_vector(x):

    # nan values
    if np.isnan(x).sum() > 0:
        imp = np.mean(x[~np.isnan(x)])
        x = np.where(np.isnan(x), imp, x)  # TODO temporary solution, should be revised
    # extreme dsc values
    if (abs(x) >= 10 ** 25).sum() > 0:
        imp = np.mean(x[abs(x) <= 10 ** 25])
        x = np.where(abs(x) <= 10 ** 25, x, imp)
    return x

class RDKitDescriptor3D:
    def __init__(self, desc_name=None):
        super().__init__()

        if desc_name:
            self.transformer = getattr(Descriptors3D.rdMolDescriptors, desc_name)

    def __call__(self, mol, conformer_id=None):
        x = np.array(self.transformer(mol, confId=conformer_id))
        x = validate_desc_vector(x)
        return x

class RDKitGEOM(RDKitDescriptor3D):
    def __init__(self):
        super().__init__()

        self.columns = ['CalcAsphericity',
                        'CalcEccentricity',
                        'CalcInertialShapeFactor',
                        'CalcNPR1',
                        'CalcNPR2',
                        'CalcPMI1',
                        'CalcPMI2',
                        'CalcPMI3',
                        'CalcRadiusOfGyration',
                        'CalcSpherocityIndex',
                        'CalcPBF']

    def __call__(self, mol, conformer_id=None):
        x = []
        for desc_name in self.columns:
            transformer = getattr(Descriptors3D.rdMolDescriptors, desc_name)
            x.append(transformer(mol, confId=conformer_id))
        x = np.array(x)
        x = validate_desc_vector(x)
        return x


class RDKitAUTOCORR(RDKitDescriptor3D):
    def __init__(self):
        super().__init__('CalcAUTOCORR3D')


class RDKitRDF(RDKitDescriptor3D):
    def __init__(self):
        super().__init__('CalcRDF')


class RDKitMORSE(RDKitDescriptor3D):
    def __init__(self):
        super().__init__('CalcMORSE')


class RDKitWHIM(RDKitDescriptor3D):
    def __init__(self):
        super().__init__('CalcWHIM')


class RDKitGETAWAY(RDKitDescriptor3D):
    def __init__(self):
        super().__init__('CalcGETAWAY')
