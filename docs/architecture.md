# Code and project boundaries

The application layer resolves explicit model/project inputs, then calls the
scientific core. Core modules depend on the neutral backend, which delegates
metadata and calculation to DeepMD. Legacy adapter imports alias the neutral
modules to preserve monkeypatch and mutable-state identity. The application
entrypoint may import the adapter; scientific core modules do not.

Scientific mechanisms belong to code: optimization, constraints, seed validation,
projection checks, channel enumeration, aggregation and data validation. User
inputs select checkpoint/head, structures/seeds, thresholds, workspace/output,
interpreter and job resources. No checkpoint selection is inferred from a filename
or an S/M label. Job accounts and resources belong to explicit deployment inputs.

Historical Fe5C2 data and wave/report rules remain named project compatibility
features. Generalizing their scientific meaning is not accomplished by moving
files or changing a title. Tests, evidence and private project data are excluded from the wheel.
The finalizer shell is explicitly packaged with its Python tool consumers.

The installed builder must load geometry from the installed core and write into
its invocation directory by default. Tool wrappers, canonical modules and their
cross-group dependencies are shipped together. Runtime configuration returns
with its campaign consumers and focused tests; it is not required by the backend.

Acceptance checks cover model/head semantics and CPU native parity, fixed PR-4
seed references, gate decisions/provenance, dissociated-site routing, tool output
and import compatibility and installed CLI behavior.
Hosted CI and production job acceptance are separate external milestones.
