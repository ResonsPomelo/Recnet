"""
DPWorkflow — orchestrator that delegates each stage to the appropriate handler module.
"""
from __future__ import annotations

from .context import WorkflowContext
from . import adsorption
from . import ts
from . import irc
from . import energy


class DPWorkflow:
    """Top-level workflow facade.

    Delegates to handler classes for adsorption, TS, IRC, and energy stages.
    """

    def __init__(self, **kwargs):
        self.ctx = WorkflowContext(**kwargs)
        self.adsorption_handler = adsorption.AdsorptionHandler(self.ctx)
        self.ts_handler = ts.TSHandler(self.ctx)
        self.irc_handler = irc.IRCHandler(self.ctx)
        self.energy_handler = energy.EnergyHandler(self.ctx)

    # Proxy context attributes so code that accesses self.xxx still works
    def __getattr__(self, name):
        if hasattr(self.ctx, name):
            return getattr(self.ctx, name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    # ---- Stage methods --------------------------------------------------
    def generate_initial_adsorbate_guesses(self):
        self.adsorption_handler.run()

    def generate_rxn_ts_guesses_ccqn(self):
        self.ts_handler.run()

    def get_final_state_IRC(self):
        return self.irc_handler.run()

    def get_final_state_energy(self):
        return self.energy_handler.run_final_state_energy()

    def get_ads_energy(self):
        return self.energy_handler.run_adsorption_energy()


__all__ = ["DPWorkflow"]
