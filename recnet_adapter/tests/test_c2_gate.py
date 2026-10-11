from __future__ import annotations

from recnet_adapter.tools.c2_gate import qc_index


def test_qc_index_uses_case_path_not_display_name() -> None:
    """Campaign and retry chunks may reuse names such as S-c0/M-c0."""
    payload = {
        "cases": [
            {
                "case": "S-c0",
                "case_dir": "/run/campaign/S-c0",
                "ts_records": [
                    {"rxn_idx": 0, "site": 1, "vg": "0", "qc": {"qc_pass": True}}
                ],
            },
            {
                "case": "S-c0",
                "case_dir": "/run/round2/S-c0",
                "ts_records": [
                    {"rxn_idx": 0, "site": 1, "vg": "0", "qc": {"qc_pass": False}}
                ],
            },
        ]
    }

    indexed = qc_index(payload)
    assert indexed[("/run/campaign/S-c0", 0, 1, "0")]["qc_pass"] is True
    assert indexed[("/run/round2/S-c0", 0, 1, "0")]["qc_pass"] is False
