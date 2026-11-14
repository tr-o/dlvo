"""DLVO interaction for identical spheres; all model inputs use SI units."""

from dataclasses import dataclass
import math

K_B = 1.380649e-23  # J/K
E_CHARGE = 1.602176634e-19  # C
N_A = 6.02214076e23  # mol^-1
EPSILON_0 = 8.8541878128e-12  # F/m


@dataclass(frozen=True)
class Parameters:
    """Constant-potential, linear PB / Derjaguin model of identical spheres.

    ionic_strength_mol_l is ionic strength, NOT total ion concentration.
    For a 1:1 salt, it equals the salt concentration in mol/L.
    """

    radius_m: float = 100e-9
    potential_v: float = 20e-3
    ionic_strength_mol_l: float = 1e-3
    hamaker_j: float = 1e-20
    temperature_k: float = 298.15
    relative_permittivity: float = 78.5

    def __post_init__(self):
        for name, value in vars(self).items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name != "potential_v" and value <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def thermal_energy_j(self):
        return K_B * self.temperature_k

    @property
    def permittivity(self):
        return EPSILON_0 * self.relative_permittivity

    @property
    def debye_length_m(self):
        return math.sqrt(
            self.permittivity * self.thermal_energy_j
            / (2 * N_A * E_CHARGE**2 * 1000 * self.ionic_strength_mol_l)
        )

    def energy_j(self, separation_m):
        """Return (vdW, double layer, total) in J; separation is surface gap.

        Requires h << radius, Debye length << radius, and low surface
        potential |e psi / kBT| << 1. Zero energy is at infinite separation.
        """
        if not math.isfinite(separation_m) or separation_m <= 0:
            raise ValueError("surface separation must be finite and positive")
        attraction = -self.hamaker_j * self.radius_m / (12 * separation_m)
        repulsion = (
            2 * math.pi * self.permittivity * self.radius_m * self.potential_v**2
            * math.log1p(math.exp(-separation_m / self.debye_length_m))
        )
        return attraction, repulsion, attraction + repulsion
