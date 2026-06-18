"""
LLM-as-Judge evaluation for agent trajectories.

Evaluates experiment trajectory files against benchmark ``evaluate.json``
using an external LLM (Claude API) to produce per-task dimension scores
and an overall 0--5 integer rating.
"""
