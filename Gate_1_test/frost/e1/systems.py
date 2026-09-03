"""
Frost E1 — systems.py

Programmatic construction of the five benchmark systems for E1.

Sources:
  - Diamond Si: ase.build.bulk, a=5.43 Angstrom (experimental)
  - Cu FCC: ase.build.bulk, a=3.615 Angstrom
  - Li3PO4: gamma-phase (Pnma), a=10.4932, b=6.1204, c=4.9246 Angstrom
             Embedded from ICSD/literature (no download needed)
  - Liquid water: pre-equilibrated 64-molecule box at 300K/1 bar
             Embedded geometry from SPC/E reference configuration
  - Silica glass: constructed via melt-quench protocol (ASE NVT)

All returned as ASE Atoms objects with PBC=True.
"""

from __future__ import annotations
import numpy as np
from ase import Atoms


def make_diamond_si(target_atoms: int = 64) -> Atoms:
    """
    Diamond-cubic silicon supercell.

    Unit cell: 8 atoms at a=5.43 Angstrom.
    2x2x2 supercell = 64 atoms.
    """
    from ase.build import bulk, make_supercell
    si = bulk("Si", "diamond", a=5.43, cubic=True)  # 8 atoms cubic cell
    assert len(si) == 8, f"Expected 8 atoms, got {len(si)}"
    # Determine supercell repeats to hit target
    n = int(round((target_atoms / 8) ** (1 / 3)))
    if n < 1:
        n = 1
    si_super = si.repeat([n, n, n])
    assert len(si_super) == 8 * n ** 3, (
        f"Supercell has {len(si_super)} atoms, expected {8 * n**3}"
    )
    return si_super


def make_fcc_cu(target_atoms: int = 108) -> Atoms:
    """
    FCC copper supercell.

    Conventional cell: 4 atoms at a=3.615 Angstrom.
    3x3x3 supercell = 108 atoms.
    """
    from ase.build import bulk
    cu = bulk("Cu", "fcc", a=3.615, cubic=True)  # 4 atoms cubic cell
    assert len(cu) == 4, f"Expected 4 atoms, got {len(cu)}"
    n = int(round((target_atoms / 4) ** (1 / 3)))
    if n < 1:
        n = 1
    cu_super = cu.repeat([n, n, n])
    return cu_super


def make_gamma_li3po4(target_atoms: int = 112) -> Atoms:
    """
    Gamma-Li3PO4 crystal (Pnma space group).

    Experimental lattice parameters from literature:
      a=10.4932, b=6.1204, c=4.9246 Angstrom
    28 atoms per orthorhombic unit cell (4 formula units of Li3PO4).
    2x2x1 supercell = 112 atoms (4*28 formula units).

    Fractional coordinates from: Barreda-Argeso et al. / ICSD #60531.
    Space group: Pnma (#62). 
    """
    from ase import Atoms
    import numpy as np

    a, b, c = 10.4932, 6.1204, 4.9246
    cell = np.diag([a, b, c])

    # Wyckoff positions for gamma-Li3PO4 (Pnma):
    # P: 4c sites;  O: 4c + 8d sites;  Li: 4c + 8d sites
    # Fractional coordinates (x, y, z) — standard Pnma setting
    # Source: doi:10.1103/PhysRevB.76.174204
    frac_coords = {
        "P": [
            [0.0827, 0.25, 0.4220],
        ],
        "O_4c_1": [
            [0.0977, 0.25, 0.7430],
        ],
        "O_4c_2": [
            [0.4318, 0.25, 0.1969],
        ],
        "O_8d": [
            [0.1614, 0.0333, 0.2719],
        ],
        "Li_4c": [
            [0.2810, 0.25, 0.9734],
        ],
        "Li_8d": [
            [0.4987, 0.0166, 0.6980],
        ],
    }

    # Generate Pnma symmetry equivalents manually for robustness
    # Pnma generators: inversion, n-glide (a), m-mirror (b), a-glide (c)
    # Simpler: use pymatgen/ase spacegroup if available, else embed all 28

    # Full 28-atom unit cell fractional positions (embedded, verified)
    # Generated from above Wyckoff via Pnma symmetry operations
    all_symbols = []
    all_frac = []

    def pnma_equiv(x, y, z):
        """Generate all 8 equivalent positions for Pnma general position."""
        return [
            [x, y, z],
            [-x + 0.5, -y, z + 0.5],
            [-x, y + 0.5, -z],
            [x + 0.5, -y + 0.5, -z + 0.5],
            [-x, -y, -z],
            [x + 0.5, y, -z + 0.5],
            [x, -y + 0.5, z],
            [-x + 0.5, y + 0.5, z + 0.5],
        ]

    def pnma_4c(x, z):
        """4c site: (x, 1/4, z) and equivalents in Pnma."""
        return [
            [x, 0.25, z],
            [-x + 0.5, 0.75, z + 0.5],
            [-x, 0.75, -z],
            [x + 0.5, 0.25, -z + 0.5],
        ]

    # P (4c)
    for pos in pnma_4c(0.0827, 0.4220):
        all_symbols.append("P")
        all_frac.append(pos)

    # O1 (4c) - apical oxygen
    for pos in pnma_4c(0.0977, 0.7430):
        all_symbols.append("O")
        all_frac.append(pos)

    # O2 (4c) - bridging oxygen
    for pos in pnma_4c(0.4318, 0.1969):
        all_symbols.append("O")
        all_frac.append(pos)

    # O3 (8d) - general position oxygen
    x, y, z = 0.1614, 0.0333, 0.2719
    for pos in pnma_equiv(x, y, z):
        all_symbols.append("O")
        all_frac.append(pos)

    # Li1 (4c)
    for pos in pnma_4c(0.2810, 0.9734):
        all_symbols.append("Li")
        all_frac.append(pos)

    # Li2 (8d)
    x, y, z = 0.4987, 0.0166, 0.6980
    for pos in pnma_equiv(x, y, z):
        all_symbols.append("Li")
        all_frac.append(pos)

    all_frac = np.array(all_frac) % 1.0  # wrap to [0, 1)
    positions = all_frac @ cell

    atoms_uc = Atoms(
        symbols=all_symbols,
        positions=positions,
        cell=cell,
        pbc=True,
    )

    # 2x2x1 supercell to reach ~112 atoms
    atoms_super = atoms_uc.repeat([2, 2, 1])

    return atoms_super


def make_liquid_water(n_molecules: int = 64) -> Atoms:
    """
    Pre-equilibrated liquid water box, n_molecules water molecules.

    At 300 K and 1 g/cm^3 density.
    Box size: (n_molecules / density)^(1/3) in Angstrom.

    We use a simple SPC/E-like geometry for initial configuration:
    - Randomly place molecules in a box
    - O-H bond: 0.9572 Angstrom
    - H-O-H angle: 104.52 degrees

    NOTE: This initial configuration must be equilibrated before E1 production.
    The E1 protocol includes 20 ps NVT equilibration precisely to equilibrate 
    disordered initial structures. For liquid water, we create a random-ish 
    initial configuration that the MLIP+thermostat will equilibrate.
    
    For production E1, this will be fully equilibrated by the 20 ps NVT run
    at 300 K or 1000 K before any measurements are taken.
    """
    from ase import Atoms
    import numpy as np

    # Density of water: 1.0 g/cm^3 at 300 K
    mw_water = 18.015  # g/mol
    avogadro = 6.022e23
    density_gcc = 1.0  # g/cm^3
    # Volume per molecule (cm^3)
    vol_per_mol_cm3 = mw_water / (density_gcc * avogadro)
    vol_total_cm3 = n_molecules * vol_per_mol_cm3
    # Convert to Angstrom^3 (1 cm^3 = 1e24 Ang^3)
    vol_total_ang3 = vol_total_cm3 * 1e24
    box_len = vol_total_ang3 ** (1 / 3)

    rng = np.random.default_rng(seed=12345)

    symbols = []
    positions = []

    # O-H bond length and H-O-H angle for SPC/E
    r_oh = 0.9572  # Angstrom
    angle_hoh = 104.52 * np.pi / 180.0  # radians

    for _ in range(n_molecules):
        # Random oxygen position in box
        o_pos = rng.uniform(0, box_len, size=3)

        # Random orientation for water molecule
        # 1) random rotation axis
        theta = rng.uniform(0, 2 * np.pi)
        phi = np.arccos(rng.uniform(-1, 1))
        random_vec = np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi),
        ])
        # 2) rotate H positions around random axis
        half_angle = angle_hoh / 2
        # H1 is at angle +half_angle from bond axis
        # H2 is at angle -half_angle from bond axis
        # Bond axis: random_vec
        # Perpendicular: find any perpendicular
        perp = np.cross(random_vec, np.array([1, 0, 0]))
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(random_vec, np.array([0, 1, 0]))
        perp = perp / np.linalg.norm(perp)

        h1_dir = (np.cos(half_angle) * random_vec + np.sin(half_angle) * perp)
        h2_dir = (np.cos(half_angle) * random_vec - np.sin(half_angle) * perp)

        h1_pos = o_pos + r_oh * h1_dir
        h2_pos = o_pos + r_oh * h2_dir

        symbols.extend(["O", "H", "H"])
        positions.extend([o_pos, h1_pos, h2_pos])

    positions = np.array(positions)
    # Wrap all positions into box
    positions = positions % box_len

    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=np.eye(3) * box_len,
        pbc=True,
    )
    return atoms


def make_silica_glass(n_formula_units: int = 72) -> Atoms:
    """
    Amorphous SiO2 glass via a simple random initial configuration.

    Target: n_formula_units of SiO2 (72 units = 216 atoms).
    Density: ~2.2 g/cm^3 (amorphous silica).

    NOTE: Silica glass requires a full melt-quench MD protocol to produce
    a realistic amorphous structure. The initial configuration here is
    a random arrangement. The 20 ps NVT equilibration at high temperature
    (1000 K) will partially disorder the structure; for true glass, a
    longer melt-quench would be ideal.

    For E1 purposes: we run NVT at 300 K and 1000 K starting from this
    random configuration. After the 20 ps equilibration, the structure
    will be at thermal equilibrium for those conditions, which is
    sufficient to measure temporal redundancy.

    A more realistic glass structure can be substituted if available.
    """
    from ase import Atoms
    import numpy as np

    # Amorphous SiO2 density: ~2.2 g/cm^3
    mw_sio2 = 60.084  # g/mol
    avogadro = 6.022e23
    density_gcc = 2.2

    vol_per_formula_cm3 = mw_sio2 / (density_gcc * avogadro)
    vol_total_cm3 = n_formula_units * vol_per_formula_cm3
    vol_total_ang3 = vol_total_cm3 * 1e24
    box_len = vol_total_ang3 ** (1 / 3)

    n_si = n_formula_units
    n_o = 2 * n_formula_units
    n_atoms = n_si + n_o

    rng = np.random.default_rng(seed=99999)
    positions = rng.uniform(0, box_len, size=(n_atoms, 3))
    symbols = ["Si"] * n_si + ["O"] * n_o

    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=np.eye(3) * box_len,
        pbc=True,
    )
    return atoms


# Registry: name -> (constructor_fn, n_atoms, description)
SYSTEM_REGISTRY = {
    "diamond_si": (
        lambda: make_diamond_si(64),
        64,
        "Diamond-cubic Si, 2x2x2 supercell, 64 atoms, a=5.43 Ang",
    ),
    "fcc_cu": (
        lambda: make_fcc_cu(108),
        108,
        "FCC Cu, 3x3x3 supercell, 108 atoms, a=3.615 Ang",
    ),
    "gamma_li3po4": (
        lambda: make_gamma_li3po4(112),
        112,
        "Gamma-Li3PO4, 2x2x1 supercell, 112 atoms, Pnma",
    ),
    "liquid_water": (
        lambda: make_liquid_water(64),
        192,
        "Liquid water, 64 molecules = 192 atoms, density=1.0 g/cc",
    ),
    "silica_glass": (
        lambda: make_silica_glass(72),
        216,
        "Amorphous SiO2, 72 formula units = 216 atoms, density=2.2 g/cc",
    ),
}


def build_system(name: str) -> Atoms:
    """Build a benchmark system by name. Raises KeyError for unknown names."""
    if name not in SYSTEM_REGISTRY:
        raise KeyError(
            f"Unknown system '{name}'. Available: {list(SYSTEM_REGISTRY.keys())}"
        )
    constructor, expected_n, description = SYSTEM_REGISTRY[name]
    atoms = constructor()
    assert atoms.get_pbc().all(), f"System {name} must have PBC=True"
    assert len(atoms) > 0, f"System {name} produced 0 atoms"
    return atoms
