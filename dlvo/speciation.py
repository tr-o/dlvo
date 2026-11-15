"""Ideal association sensitivity for a symmetric monovalent salt only."""

import math


def free_ionic_strength_m(nominal_salt_m, association_l_mol):
    """I=c_free for neutral ion pairing, c_total=c_free+K*c_free^2.

    K is an input, not a measured ethanol or BaTiO3 constant. Activities,
    multivalent ions, surface adsorption and dissociation equilibria are omitted.
    """
    if (not math.isfinite(nominal_salt_m) or nominal_salt_m < 0
            or not math.isfinite(association_l_mol) or association_l_mol < 0):
        raise ValueError("require finite nonnegative concentration and association constant")
    return 2*nominal_salt_m/(1+math.sqrt(1+4*association_l_mol*nominal_salt_m))
