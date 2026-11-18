"""Urn Exchange: a zero-sum trading game for language models, with exact Bayesian grading.

Modules: ``exchange`` (the game and its scripted bots), ``urns`` (exact urn maths), ``core`` (the game
contract and match loop), ``runner`` (model players, tournaments, retries, cache, resume, spend guard),
``schedule`` (config and seat rotations), ``ratings`` (chips, Bradley-Terry Elo, bootstrap), ``summary``,
``traces``, ``play`` and ``cli``. Paper chips, research only, not investment advice.
"""

__version__ = "1.0.0"
