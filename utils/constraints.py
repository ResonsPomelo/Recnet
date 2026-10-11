import numpy as np
from ase.constraints import FixAtoms, FixInternals
from ase.calculators import calculator
from ase.geometry import get_distances
from recnet_backend.calculator import DP


def get_energy_forces_atom_bond(atoms, ind1, ind2, k, deq):
    forces = np.zeros(atoms.positions.shape)
    bd, d = get_distances([atoms.positions[ind1]], [atoms.positions[ind2]], cell=atoms.cell, pbc=atoms.pbc)
    if d != 0.0:
        forces[ind1] = 2.0 * bd * (1.0 - deq / d)
        forces[ind2] = -forces[ind1]
    else:
        forces[ind1] = bd
        forces[ind2] = bd
    energy = k * (d - deq) ** 2
    return energy, k * forces


def get_energy_forces_site_bond(atoms, ind, site_pos, k, deq):
    forces = np.zeros(atoms.positions.shape)
    bd, d = get_distances([atoms.positions[ind]], [site_pos], cell=atoms.cell, pbc=atoms.pbc)
    if d != 0:
        forces[ind] = 2.0 * bd * (1.0 - deq / d)
    else:
        forces[ind] = bd
    energy = k * (d - deq) ** 2
    return energy, k * forces


def get_energy_forces_center_bond(atoms, indices, site_pos, k, deq):
    """Harmonic restraint between adsorbate center and adsorption site."""
    forces = np.zeros(atoms.positions.shape)
    if indices is None or len(indices) == 0:
        return 0.0, forces

    center = np.mean(atoms.positions[indices], axis=0)
    bd, d = get_distances([center], [site_pos], cell=atoms.cell, pbc=atoms.pbc)

    if d != 0:
        f_center = 2.0 * bd * (1.0 - deq / d)
    else:
        f_center = bd

    f_each = f_center / float(len(indices))
    for idx in indices:
        forces[idx] += f_each

    energy = k * (d - deq) ** 2
    return energy, k * forces


class HarmonicallyForcedDP(DP):
    def get_energy_forces(self):
        energy = 0.0
        forces = np.zeros(self.atoms.positions.shape)
        if hasattr(self.parameters, "atom_bond_potentials"):
            for atom_bond_potential in self.parameters.atom_bond_potentials:
                e_add, f_add = get_energy_forces_atom_bond(self.atoms, **atom_bond_potential)
                energy += e_add
                forces += f_add

        if hasattr(self.parameters, "site_bond_potentials"):
            for site_bond_potential in self.parameters.site_bond_potentials:
                e_add, f_add = get_energy_forces_site_bond(self.atoms, **site_bond_potential)
                energy += e_add
                forces += f_add

        if hasattr(self.parameters, "center_bond_potentials"):
            for center_bond_potential in self.parameters.center_bond_potentials:
                e_add, f_add = get_energy_forces_center_bond(self.atoms, **center_bond_potential)
                energy += e_add
                forces += f_add

        if not isinstance(energy, float):
            energy = energy[0][0]
        return energy, forces

    def calculate(self, atoms=None, properties=None, system_changes=calculator.all_changes):
        DP.calculate(self, atoms=atoms, properties=properties, system_changes=system_changes)
        e_add, f_add = self.get_energy_forces()
        self.results["energy"] += e_add
        self.results["free_energy"] += e_add
        self.results["forces"] += f_add


def build_constraints(atoms, normal_axis, bottom_freeze_threshold, surf_atom_num=0, with_internal_bonds=False):
    axis_id = {"x": 0, "y": 1, "z": 2}[normal_axis]
    frozen_indices = [
        atom.index
        for atom in atoms
        if float(atom.position[axis_id]) < float(bottom_freeze_threshold)
    ]

    constraints = [FixAtoms(indices=frozen_indices)]
    if with_internal_bonds:
        from WG.code.utils.geometry import get_bond_connections

        bonds = get_bond_connections(atoms, shift=surf_atom_num, bond_type="nosurf")
        _, distances = get_distances(atoms.get_positions(), None, atoms.get_cell(), [True, True, True])
        new_bonds = []
        for bond in bonds:
            new_bond = [bond[0] + surf_atom_num, bond[1] + surf_atom_num]
            dis = distances[new_bond[0], new_bond[1]]
            new_bonds.append([dis, new_bond])
        constraints.insert(0, FixInternals(bonds=new_bonds))
    return constraints
