"""Coverage of the ``network-Fe5C2-0813`` paper steps (parent supplement)."""

from __future__ import annotations

import yaml

from network_builder.build import PAPER_STEPS


def _paper_index(manifest):
    paper = manifest["paper_coverage"]
    index = {}
    for entry in paper["mapped"]:
        index[entry["id"]] = entry
    for entry in paper["boundaries"]:
        index[entry["id"]] = entry
    for entry in paper["unresolved"]:
        index[entry["id"]] = entry
    return paper, index


def test_every_paper_step_is_classified(manifest):
    paper, index = _paper_index(manifest)
    assert paper["verification_failures"] == []
    assert len(index) == len(PAPER_STEPS)
    for step in PAPER_STEPS:
        assert step["status"] in {"mapped", "mapped_with_cv_caveat", "partial",
                                  "boundary_cv", "unresolved"}


def test_mapped_steps_really_exist_in_the_generated_yaml(manifest):
    """Each mapped paper step must be present as a concrete yaml reaction."""
    payload = yaml.safe_load(open(manifest["_yaml_path"], encoding="utf-8"))
    sp_by_key = {entry["key"]: entry["sp_id"] for entry in manifest["species_entries"]}
    existing = {
        (
            tuple(rxn["reactant_species"]),
            tuple(rxn["product_species"]),
        )
        for rxn in payload["rxns"]
    }
    mapped_count = 0
    for entry in manifest["paper_coverage"]["mapped"]:
        channel = entry["channel"]
        assert channel is not None, f"{entry['id']} is marked mapped but has no channel"
        pair = ((sp_by_key[channel["reactant"]],),
                tuple(sp_by_key[key] for key in channel["products"]))
        assert pair in existing, f"{entry['id']} channel missing from the yaml"
        mapped_count += 1
    assert mapped_count >= 13


def test_boundary_steps_explain_the_unsupported_physics(manifest):
    boundaries = manifest["paper_coverage"]["boundaries"]
    assert boundaries, "the Cv-coupled paper steps must be listed as boundaries"
    for entry in boundaries:
        assert "Cv" in entry["paper"] or "空位" in entry["reason"]
        assert entry["suggestion"], "a boundary needs a concrete suggestion"
        assert "channel" not in entry  # never forced into the schema


def test_unresolved_step_is_flagged_with_candidates(manifest):
    unresolved = manifest["paper_coverage"]["unresolved"]
    assert len(unresolved) == 1
    entry = unresolved[0]
    assert entry["id"] == "CCH3_CCCH3"
    assert entry["candidates"] and all(item["present"] for item in entry["candidates"])
    assert entry["reason"] and entry["suggestion"]


def test_increment_counts_split_paper_covered_and_new_channels(manifest):
    """Increment evidence: channels the paper network does not contain."""
    paper = manifest["paper_coverage"]
    increments = paper["increment_channels_by_bond_type"]
    assert increments, "increment breakdown must be reported"
    total_increment = sum(increments.values())
    assert total_increment == manifest["counts"]["rxns"] - paper["paper_covered_channel_count"]
    assert len(paper["increment_channels"]) == total_increment
    assert set(increments) <= {
        "C-H (order 1)", "C-H (order 2)", "C-H (order 3)",
        "C-C (order 1)", "C-C (order 2)", "C-C (order 3)",
        "C-O (order 1)", "C-O (order 2)", "C-O (order 3)",
        "O-H (order 1)", "H-H (order 1)",
    }


def test_channel_classes_cover_the_desorption_direction(manifest):
    classes = manifest["paper_coverage"]["channel_classes"]
    assert classes["gas_entry_scission_desorption_direction"] == 5
    assert classes["surface_scission"] + classes["gas_entry_scission_desorption_direction"] \
        == manifest["counts"]["rxns"]
    assert classes["pure_desorption_without_bond_break"] == 0
    assert "energy.py" in classes["note"]


def test_paper_step_inventory_matches_the_parent_list(manifest):
    """The supplement names 17 steps; the local archive adds one more."""
    paper, index = _paper_index(manifest)
    expected_anchors = {
        "COdis", "TS1_2H_to_CHCv_H", "TS2_CHCv_H_to_CH_Cv_H",
        "TS3_CH_CO_H_to_CH_CCv_O_H", "C-CH", "C-CH4/CH3-H", "C-CH4/CH2-H",
        "C-CH4/CH-H", "C-CH4/sur-H", "CH-CH", "TS5_TS6_CCH_H_to_CCH2",
        "TS8_CCH2_H_to_CCH3", "CCH2_CHCH2", "CCH3_CCCH3", "CCH3_Cv",
        "CHCH_CHCH2", "CHCH2_CH2CH2", "CHCH2_Cv",
    }
    assert set(index) == expected_anchors
    assert len(paper["boundaries"]) == 4
