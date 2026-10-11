from ase.visualize import view
from ase.io import write, read
import numpy as np
from ase import Atoms
from scipy.spatial import Voronoi
import os

import math
from copy import deepcopy
from ase.geometry import get_distances


def _maybe_view(stru, show):
    """Opt-in Voronoi viewer.

    2026-09-21: `ase.visualize.view()` spawns `python -m ase gui -`.  Every code
    path that builds a SlabSite with ``show=True`` (pipeline, tests, campaigns)
    used to spawn a viewer: with DISPLAY set it pops windows, without DISPLAY it
    still forks a doomed GUI process per call (invisible in logs).  The viewer is
    therefore strictly opt-in via ``SLABSITE_SHOW_VORONOI=1``.
    """
    if show and os.environ.get("SLABSITE_SHOW_VORONOI") == "1":
        view(stru)

def _is_point_in_parallelepiped(p, a, b, c, origin=np.array([0, 0, 0])):
    """
    判断一个点是否处于由三个向量组成的六面体内。

    参数:
    - p: ndarray, 点的坐标 (x, y, z)。
    - origin: ndarray, 六面体的起点 (原点)。
    - a, b, c: ndarray, 六面体的三个向量。

    返回:
    - bool: 如果点在六面体内，返回 True；否则返回 False。
    """
    # 将点转换到局部坐标系
    p_local = p - origin

    # 构造矩阵，将六面体的三个向量作为列向量
    matrix = np.column_stack((a, b, c))

    # 求解线性方程组，计算 u, v, w
    try:
        u, v, w = np.linalg.solve(matrix, p_local)
    except np.linalg.LinAlgError:
        # 如果矩阵不可逆，返回 False
        return False

    # 检查 u, v, w 是否在 [0, 1] 范围内
    return 0 <= u < 1 and 0 <= v < 1 and 0 <= w < 1


class SlabSite():
    def __init__(self, slab_path, element='Fe', z_min=6.24, normal_axis='z'):
        self.stru = read(slab_path)
        self.element = element
        self.z_min = z_min
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        axis_key = str(normal_axis).lower()
        if axis_key not in axis_map:
            raise ValueError("normal_axis must be one of 'x', 'y', 'z'")
        self.normal_axis = axis_key
        self.normal_idx = axis_map[axis_key]
        self.plane_axes = [i for i in [0, 1, 2] if i != self.normal_idx]
        self._pbc = tuple(i in self.plane_axes for i in [0, 1, 2])

    def expand_slab(self):
        stru = self.stru
        element = self.element
        z_min = self.z_min

        repeat_vec = [1, 1, 1]
        for axis in self.plane_axes:
            repeat_vec[axis] = 3
        stru = stru.repeat(tuple(repeat_vec))

        if z_min is None:
            z_min = np.max(stru.positions[:, self.normal_idx]) - 0.6

        top_layer_indices = np.array(
            [atom.index for atom in stru if atom.position[self.normal_idx] > z_min and atom.symbol == element],
            dtype=int,
        )
        if top_layer_indices.size == 0:
            raise ValueError("No atoms satisfy z_min/element filter; check z_min or element settings.")
        top_layer = stru[top_layer_indices]

        # view(top_layer)

        return stru, top_layer
    
    def post_process(self, stru, vertices):
        a, b, c = self.stru.cell

        shift_vec = self.stru.cell[self.plane_axes[0]] + self.stru.cell[self.plane_axes[1]]

        stru.positions -= shift_vec

        stru.set_cell(self.stru.cell)

        mask = []
        for pos in stru.positions:
            mask.append(_is_point_in_parallelepiped(pos, a, b, c))

        # 利用mask过滤掉不在六面体内的原子
        stru = stru[mask]
        
        vertices -= shift_vec

        mask = []
        for pos in vertices:
            mask.append(_is_point_in_parallelepiped(pos, a, b, c))

        vertices = vertices[mask]
        
        return stru, vertices

    def voronoi(self, show=False, type=["hollow"]):
        # show = False
        stru, top_layer = self.expand_slab()

        vor = Voronoi(top_layer.positions[:, self.plane_axes])
        hollow_sites = vor.vertices

        vertices = hollow_sites

        if 'bridge' in type:
            ridge_vertices = vor.ridge_vertices
            bridge_sites = []
            for ridge in ridge_vertices:
                if -1 in ridge:  # 跳过无穷远的边
                    continue
                v1, v2 = hollow_sites[ridge]
                midpoint = (v1 + v2) / 2  # 计算中点
                bridge_sites.append(midpoint)
            bridge_sites = np.array(bridge_sites)
            vertices = np.append(vertices, bridge_sites, axis=0)
        if 'top' in type:
            top_sites = top_layer.positions[:, self.plane_axes]
            vertices = np.append(vertices, top_sites, axis=0)

        z_max = np.max(top_layer.positions[:, self.normal_idx])
        full_vertices = np.zeros((vertices.shape[0], 3))
        full_vertices[:, self.plane_axes[0]] = vertices[:, 0]
        full_vertices[:, self.plane_axes[1]] = vertices[:, 1]
        full_vertices[:, self.normal_idx] = z_max + 0.5
        vertices = full_vertices

        if show:
            site = Atoms('H' * len(vertices), positions=vertices)
            stru.extend(site)

        stru, sites = self.post_process(stru, vertices)

        self.sites = sites

        _maybe_view(stru, show)

    def filter_sites_near(self, threshold=0.6):
        sites = self.sites

        # stru = self.stru
        # stru.extend(Atoms('H' * len(sites), positions=sites))
        # view(stru)

        del_list = set()
        for i, site in enumerate(sites):
            for j, other_site in enumerate(sites):
                if i >= j:
                    continue
                _, distance = get_distances([site], [other_site], cell=self.stru.cell, pbc=self._pbc)
                if distance[0][0] < threshold:
                    del_list.add(j)
        
        sites = np.delete(sites, list(del_list), axis=0)
        self.sites = sites


        # stru = self.stru
        # stru.extend(Atoms('H' * len(sites), positions=sites))
        # view(stru)

    def filter_unique_site(self):
        self.filter_sites_near()

        stru = self.stru
        z_min = self.z_min if self.z_min is not None else np.max(stru.positions[:, self.normal_idx]) - 0.6
        top_layer_indices = [atom.index for atom in stru if atom.position[self.normal_idx] > z_min and atom.symbol == self.element]
        top_layer = stru[top_layer_indices]

        sites = self.sites

        # 获取site与所有top_layer原子的距离，使用get_distances
        _, distances = get_distances(sites, top_layer.positions, cell=stru.cell, pbc=self._pbc)
        # print(distances.shape)

        # sort the distances
        distances.sort(axis=1)

        # 对比每个site与top_layer原子的距离，如果完全相同，则认为是同一个site
        unique_sites = {
            'idx':[],
            'pos':[]
        }
        sites_type = {}
        for i, site in enumerate(sites):
            unique = True
            for j, site2 in enumerate(sites):
                if j >= i:
                    continue
                d = distances[i] - distances[j]
                d = np.sum(np.abs(d))
                if d < 1e-3:
                    unique = False
                    sites_type[j].append(i)
                    break
            if unique:
                unique_sites['idx'].append(i)
                unique_sites['pos'].append(site)
                sites_type[i] = []
                sites_type[i].append(i)

        self.unique_sites = unique_sites
        self.sites_type = sites_type

        # print(len(unique_sites['idx']))
        # print(sites_type)

        # stru = self.stru
        # stru.extend(Atoms('H' * len(unique_sites['pos']), positions=unique_sites['pos']))
        # view(stru)

    def find_pairs(self, radius=3, neighbors_num=2, min_dist=1.0):
        sites = self.sites
        unique_sites = self.unique_sites['pos']
        unique_sites_idx = self.unique_sites['idx']

        pairs = {}
        all_pairs = []

        for i in unique_sites_idx:
            site = sites[i]
            pairs[i] = []
            distances = []
            for j, other_site in enumerate(sites):
                if i == j:
                    continue
                distance = get_distances([site], [other_site], cell=self.stru.cell, pbc=self._pbc)[1][0][0]
                if distance < radius and distance > min_dist:
                    pairs[i].append(j)
                    all_pairs.append((i, j))
                distances.append((distance, j))

            if len(pairs[i]) < neighbors_num:
                distances.sort()
                for distance, j in distances[:neighbors_num]:
                    if (i, j) not in pairs[i] and distance > min_dist:
                        pairs[i].append(j)
                        all_pairs.append((i, j))
    
        self.pairs = pairs
        self.all_pairs = all_pairs

        # print(len(all_pairs))
        # print(len(pairs))
        # print(all_pairs)

    def get_site(self, idx):
        return self.sites[idx]
    
    def get_distance(self, idx1, idx2):
        pos1 = self.get_site(idx1)
        pos2 = self.get_site(idx2)

        from ase.geometry import get_distances

        _, distance = get_distances([pos1], [pos2], cell=self.stru.cell, pbc=self._pbc)

        return distance[0][0]

    def _get_surface_indices(self, element='C', z_min=None):
        stru = self.stru
        element_indices = [atom.index for atom in stru if atom.symbol == element]
        if len(element_indices) == 0:
            raise ValueError(f"No atoms with element {element} found in structure.")

        if z_min is None:
            # Use element-specific top layer reference; global top atoms may belong to
            # another element and can over-filter target vacancy atoms.
            element_heights = stru.positions[element_indices, self.normal_idx]
            z_min = float(np.max(element_heights) - 0.6)

        surface_indices = [
            atom.index
            for atom in stru
            if atom.symbol == element and atom.position[self.normal_idx] > z_min
        ]
        return surface_indices, z_min

    def _vacancy_fingerprint(self, atom_index, cutoff=4.0, precision=3):
        center = self.stru.positions[atom_index]
        symbols = self.stru.get_chemical_symbols()
        _, distances = get_distances([center], self.stru.positions, cell=self.stru.cell, pbc=self._pbc)
        distances = distances[0]

        env = []
        for i, d in enumerate(distances):
            if i == atom_index:
                continue
            if d <= cutoff:
                env.append((symbols[i], round(float(d), precision)))

        env.sort()
        return tuple(env)

    def enumerate_unique_surface_vacancies(self, vacancy_element='C', z_min=None, cutoff=4.0, precision=3):
        surface_indices, z_min_used = self._get_surface_indices(vacancy_element, z_min)
        if len(surface_indices) == 0:
            raise ValueError(
                f"No surface {vacancy_element} atoms found. Try lowering z_min (current: {z_min_used:.3f})."
            )

        grouped = {}
        for idx in surface_indices:
            fp = self._vacancy_fingerprint(idx, cutoff=cutoff, precision=precision)
            grouped.setdefault(fp, []).append(idx)

        unique_groups = []
        for i, (_, members) in enumerate(grouped.items()):
            unique_groups.append(
                {
                    'group_id': i,
                    'representative_index': members[0],
                    'members': members,
                }
            )

        self.vacancy_groups = unique_groups
        self.vacancy_element = vacancy_element
        self.vacancy_env_params = {
            'z_min': z_min_used,
            'cutoff': cutoff,
            'precision': precision,
        }
        return unique_groups

    def build_vacancy_structure(self, atom_index, marker_symbol='H'):
        stru = deepcopy(self.stru)
        vacancy_position = np.array(stru.positions[atom_index], dtype=float)
        vacancy_symbol = stru[atom_index].symbol
        del stru[atom_index]

        if marker_symbol is not None:
            marker = Atoms(marker_symbol, positions=[vacancy_position])
            # tag=99 marks the atom as a hole-site marker for post-processing scripts.
            marker.set_tags([99])
            stru.extend(marker)

        stru.info['vacancy_symbol'] = vacancy_symbol
        stru.info['vacancy_position'] = vacancy_position.tolist()
        stru.info['hole_site_marker'] = marker_symbol
        return stru

    def save_unique_vacancy_structures(
        self,
        output_dir='c_vacancy_structures',
        vacancy_element='C',
        z_min=None,
        cutoff=4.0,
        precision=3,
        marker_symbol='H',
        write_all_members=False,
    ):
        groups = self.enumerate_unique_surface_vacancies(
            vacancy_element=vacancy_element,
            z_min=z_min,
            cutoff=cutoff,
            precision=precision,
        )

        os.makedirs(output_dir, exist_ok=True)
        saved = []

        for g in groups:
            rep = g['representative_index']
            stru_vac = self.build_vacancy_structure(rep, marker_symbol=marker_symbol)
            file_name = f"{vacancy_element}_vac_unique_{g['group_id']:03d}_rep{rep}.cif"
            file_path = os.path.join(output_dir, file_name)
            write(file_path, stru_vac)
            saved.append((file_path, g['members']))

            if write_all_members:
                for member_idx in g['members']:
                    stru_member = self.build_vacancy_structure(member_idx, marker_symbol=marker_symbol)
                    member_name = (
                        f"{vacancy_element}_vac_group{g['group_id']:03d}_idx{member_idx}.cif"
                    )
                    member_path = os.path.join(output_dir, member_name)
                    write(member_path, stru_member)

        self.saved_vacancy_files = saved
        return saved


if __name__ == "__main__":

    path = './Fe5C2 (1 1 1).cif'
    slab = SlabSite(path, normal_axis='y', z_min=None)

    slab.voronoi(False, ["hollow"])
    slab.filter_unique_site()
    slab.voronoi(False, ["top", "bridge", "hollow"])
    
    print(f"Unique sites: {len(slab.unique_sites['pos'])}")
    print(slab.unique_sites['idx'])

    # Enumerate chemically unique surface C vacancies, mark hole site with H, and save.
    saved = slab.save_unique_vacancy_structures(
        output_dir='c_vacancy_structures',
        vacancy_element='C',
        z_min=None,
        cutoff=4.0,
        precision=3,
        marker_symbol='H',
        write_all_members=False,
    )
    print(f"Unique C-vacancy environments: {len(saved)}")
    for file_path, members in saved:
        print(f"saved: {file_path}; equivalent C indices: {members}")

    # slab.find_pairs(radius=1.5)
    # slab.show_pairs()
