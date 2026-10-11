import argparse
import os
from dataclasses import dataclass

import numpy as np

from slabsite import SlabSite


@dataclass
class WorkflowConfig:
    path: str
    slab: str
    prepared: str
    top_x: int
    enum_ads: bool
    enum_ts: bool
    surface_normal: tuple
    normal_axis: str
    imag_mode_check: bool
    imag_mode_displacement: float
    imag_mode_relax_steps: int
    bottom_freeze_threshold: float | None
    use_c_vacancy_io: bool
    vacancy_input: str | None
    vacancy_output_dir: str
    vacancy_element: str
    vacancy_group_indices: list[int] | None
    vacancy_z_min: float | None
    vacancy_cutoff: float
    vacancy_precision: int
    vacancy_marker_symbol: str | None
    vacancy_write_all_members: bool
    run_irc_final_state: bool
    irc_fmax: float
    irc_steps: int
    irc_dx: float
    irc_eta: float
    irc_ninner_iter: int
    irc_thermo_corrections: bool
    irc_temperature: float
    thermo_corrections: bool
    gas_species_whitelist: list[str] | None
    gas_pressure_pa: float
    ts_seed: bool = True


def parse_surface_normal(text):
    """Parse '--surface-normal' argument like '0,1,0' or '0 1 0'."""
    parts = text.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError("surface normal must contain 3 numbers")
    vec = tuple(float(x) for x in parts)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        raise ValueError("surface normal length is zero")
    return tuple(np.array(vec, dtype=float) / norm)


def infer_normal_axis(surface_normal):
    """Infer dominant Cartesian axis for SlabSite.normal_axis."""
    axis_names = ["x", "y", "z"]
    idx = int(np.argmax(np.abs(np.array(surface_normal, dtype=float))))
    return axis_names[idx]


def parse_marker_symbol(text):
    if text is None:
        return None
    value = str(text).strip()
    if value.lower() in {"", "none", "null"}:
        return None
    return value


def parse_vacancy_group_indices(text):
    """Parse vacancy group indices from 'all', '0', '0,2,5', or '0 2 5'."""
    if text is None:
        return [0]

    value = str(text).strip()
    if value == "":
        return [0]
    if value.lower() == "all":
        return None

    parts = value.replace(",", " ").split()
    indices = [int(x) for x in parts]
    if len(indices) == 0:
        return [0]
    return indices


def _dedupe_keep_order(items):
    seen = set()
    out = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def parse_species_whitelist(text):
    """Parse gas species whitelist from 'auto', 'all', 'sp_000,sp_003' or 'sp_000 sp_003'."""
    if text is None:
        return None

    value = str(text).strip()
    if value == "":
        return None
    if value.lower() in {"auto", "all", "none", "null"}:
        return None

    parts = value.replace(",", " ").split()
    if len(parts) == 0:
        return None
    return _dedupe_keep_order(parts)


def resolve_slab_paths_for_workflow(cfg):
    """Resolve one or more slab paths with optional C-vacancy IO handling."""
    if cfg.vacancy_input:
        if not os.path.exists(cfg.vacancy_input):
            raise FileNotFoundError(f"vacancy input not found: {cfg.vacancy_input}")
        print(f"Using vacancy input slab: {cfg.vacancy_input}")
        return [{
            "group_index": None,
            "slab_path": os.path.abspath(cfg.vacancy_input),
            "members": None,
        }]

    if not cfg.use_c_vacancy_io:
        return [{
            "group_index": None,
            "slab_path": cfg.slab,
            "members": None,
        }]

    out_dir = cfg.vacancy_output_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(cfg.path, out_dir)

    slab_for_vac = SlabSite(cfg.slab, normal_axis=cfg.normal_axis, z_min=None)
    saved = slab_for_vac.save_unique_vacancy_structures(
        output_dir=out_dir,
        vacancy_element=cfg.vacancy_element,
        z_min=cfg.vacancy_z_min,
        cutoff=cfg.vacancy_cutoff,
        precision=cfg.vacancy_precision,
        marker_symbol=cfg.vacancy_marker_symbol,
        write_all_members=cfg.vacancy_write_all_members,
    )

    if len(saved) == 0:
        raise RuntimeError("No vacancy structures were generated.")

    print(f"Generated {len(saved)} unique {cfg.vacancy_element}-vacancy groups in {out_dir}")

    if cfg.vacancy_group_indices is None:
        group_indices = list(range(len(saved)))
    else:
        group_indices = _dedupe_keep_order(cfg.vacancy_group_indices)

    selected = []
    for group_index in group_indices:
        if group_index < 0 or group_index >= len(saved):
            raise ValueError(
                f"vacancy-group-index {group_index} out of range, valid: 0..{len(saved)-1}"
            )
        selected_path, members = saved[group_index]
        selected_path = os.path.abspath(selected_path)
        print(
            f"Using vacancy group {group_index}: {selected_path}; "
            f"equivalent indices: {members}"
        )
        selected.append({
            "group_index": int(group_index),
            "slab_path": selected_path,
            "members": members,
        })

    return selected


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run DP-only adsorption and TS workflow from prepared RMG data")
    parser.add_argument("--path", default=".")
    parser.add_argument("--slab", default="./fe3c-010-0.00.poscar")
    parser.add_argument("--prepared", default="./prepared_data/prepared_rmg_data.yaml")
    parser.add_argument("--top-x", type=int, default=1, help="Only keep the lowest-energy top X structures per species")
    parser.add_argument("--enum-ads", action="store_true", help="Enable azimuthal rotation enumeration for adsorption guesses")
    parser.add_argument("--enum-ts", dest="enum_ts", action="store_true", help="Enable azimuthal rotation enumeration for TS guesses")
    parser.add_argument("--no-enum-ts", dest="enum_ts", action="store_false", help="Disable azimuthal rotation enumeration for TS guesses")
    parser.add_argument(
        "--ts-seed",
        dest="ts_seed",
        action="store_true",
        help="Use per-site TS seed files under <case>/rxn/seeds/ when present (default: enabled)",
    )
    parser.add_argument(
        "--no-ts-seed",
        dest="ts_seed",
        action="store_false",
        help="Disable per-site TS seeds; always build TS guesses from the adsorbate templates",
    )
    parser.add_argument("--surface-normal", default="0,0,1", help="Surface normal vector, e.g. '0,0,1'")
    parser.add_argument("--normal-axis", choices=["x", "y", "z"], default='z', help="Optional normal axis for SlabSite; inferred from surface normal if omitted")
    parser.add_argument("--bottom-freeze-threshold", type=float, default=None, help="Freeze atoms with coordinate below this threshold along normal axis")

    parser.add_argument("--imag-mode-check", dest="imag_mode_check", action="store_true", help="Run +/- imaginary-mode displacement endpoint checks after TS search")
    parser.add_argument("--no-imag-mode-check", dest="imag_mode_check", action="store_false", help="Disable +/- imaginary-mode displacement endpoint checks")
    parser.add_argument("--imag-mode-displacement", type=float, default=0.15, help="Displacement amplitude (Angstrom) for imaginary-mode endpoint checks")
    parser.add_argument("--imag-mode-relax-steps", type=int, default=40, help="MDMin steps for each imaginary-mode displaced endpoint")

    parser.add_argument("--use-c-vacancy-io", action="store_true", help="Generate unique C-vacancy structures and use one or more as slab input")
    parser.add_argument("--vacancy-input", default=None, help="Directly use an existing vacancy structure file as slab input")
    parser.add_argument("--vacancy-output-dir", default="c_vacancy_structures", help="Output directory for generated vacancy structures")
    parser.add_argument("--vacancy-element", default="C", help="Element to remove when generating vacancies")
    parser.add_argument(
        "--vacancy-group-index",
        default="all",
        help="Vacancy group indices to run, e.g. 'all', '0', or '0,2,5'",
    )
    parser.add_argument("--vacancy-z-min", type=float, default=None, help="Optional z_min for surface vacancy screening")
    parser.add_argument("--vacancy-cutoff", type=float, default=4.0, help="Neighbor cutoff for vacancy environment fingerprint")
    parser.add_argument("--vacancy-precision", type=int, default=3, help="Distance rounding precision for vacancy fingerprint")
    parser.add_argument("--vacancy-marker-symbol", default=None, help="Marker symbol for hole site; use 'none' to disable")
    parser.add_argument("--vacancy-write-all-members", action="store_true", help="Also write all equivalent vacancy members")

    parser.add_argument("--run-irc-final-state", action="store_true", help="Run IRC from generated TS records and search final states")
    parser.add_argument("--irc-fmax", type=float, default=0.08, help="fmax used in IRC optimization")
    parser.add_argument("--irc-steps", type=int, default=200, help="Maximum IRC steps for each direction")
    parser.add_argument("--irc-dx", type=float, default=0.1, help="IRC step size dx")
    parser.add_argument("--irc-eta", type=float, default=0.0002, help="IRC eta parameter")
    parser.add_argument("--irc-ninner-iter", type=int, default=50, help="IRC ninner_iter parameter")
    parser.add_argument(
        "--irc-thermo-corrections",
        dest="irc_thermo_corrections",
        action="store_true",
        help="Enable vibration-based ZPE/G corrections in get_final_state_IRC",
    )
    parser.add_argument(
        "--no-irc-thermo-corrections",
        dest="irc_thermo_corrections",
        action="store_false",
        help="Disable vibration-based ZPE/G corrections in get_final_state_IRC",
    )
    parser.add_argument(
        "--irc-temperature",
        type=float,
        default=573.15,
        help="Temperature (K) used for IRC vibration free-energy corrections",
    )

    parser.add_argument(
        "--thermo-corrections",
        dest="thermo_corrections",
        action="store_true",
        help="Enable thermo corrections (ZPE/G) in final-state energy evaluation",
    )
    parser.add_argument(
        "--no-thermo-corrections",
        dest="thermo_corrections",
        action="store_false",
        help="Disable thermo corrections and set all correction terms/sources to 0",
    )
    parser.add_argument(
        "--gas-species-whitelist",
        default="auto",
        help="Optional gas-species whitelist, e.g. 'sp_000,sp_003'. Use 'auto' to infer from ad_idx=[]",
    )
    parser.add_argument(
        "--gas-pressure-pa",
        type=float,
        default=101325.0,
        help="Gas pressure (Pa) for ideal-gas free energy corrections, e.g. 8.3e5 for 0.83 MPa",
    )

    parser.set_defaults(enum_ts=True)
    parser.set_defaults(ts_seed=True)
    parser.set_defaults(imag_mode_check=True)
    parser.set_defaults(use_c_vacancy_io=False)
    parser.set_defaults(run_irc_final_state=False)
    parser.set_defaults(irc_thermo_corrections=False)
    parser.set_defaults(thermo_corrections=True)
    return parser


def config_from_args(args):
    return WorkflowConfig(
        path=args.path,
        slab=args.slab,
        prepared=args.prepared,
        top_x=args.top_x,
        enum_ads=args.enum_ads,
        enum_ts=args.enum_ts,
        ts_seed=args.ts_seed,
        surface_normal=parse_surface_normal(args.surface_normal),
        normal_axis=args.normal_axis,
        imag_mode_check=args.imag_mode_check,
        imag_mode_displacement=args.imag_mode_displacement,
        imag_mode_relax_steps=args.imag_mode_relax_steps,
        bottom_freeze_threshold=args.bottom_freeze_threshold,
        use_c_vacancy_io=args.use_c_vacancy_io,
        vacancy_input=args.vacancy_input,
        vacancy_output_dir=args.vacancy_output_dir,
        vacancy_element=args.vacancy_element,
        vacancy_group_indices=parse_vacancy_group_indices(args.vacancy_group_index),
        vacancy_z_min=args.vacancy_z_min,
        vacancy_cutoff=args.vacancy_cutoff,
        vacancy_precision=args.vacancy_precision,
        vacancy_marker_symbol=parse_marker_symbol(args.vacancy_marker_symbol),
        vacancy_write_all_members=args.vacancy_write_all_members,
        run_irc_final_state=args.run_irc_final_state,
        irc_fmax=args.irc_fmax,
        irc_steps=args.irc_steps,
        irc_dx=args.irc_dx,
        irc_eta=args.irc_eta,
        irc_ninner_iter=args.irc_ninner_iter,
        irc_thermo_corrections=args.irc_thermo_corrections,
        irc_temperature=args.irc_temperature,
        thermo_corrections=args.thermo_corrections,
        gas_species_whitelist=parse_species_whitelist(args.gas_species_whitelist),
        gas_pressure_pa=args.gas_pressure_pa,
    )
