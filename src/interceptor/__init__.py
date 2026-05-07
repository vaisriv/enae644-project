"""Interceptor agent components (Agent I).

Public API (available once modules are implemented):
    LearnedRewardFunction - Neural network parameterization of learned reward
    maximum_entropy_irl   - Train IRL model from demonstrations
    predict_trajectory    - Predict trajectory under learned reward
    load_irl_model        - Load trained IRL model from checkpoint
    ParticleFilter        - Particle filter for goal inference
    game_theoretic_mpc    - Compute optimal control via game-theoretic MPC
    bayesian_update       - Perform Bayesian belief update
    compute_likelihood    - Compute P(obs | goal) under learned model
"""

# TODO: implement irl.py, particle_filter.py, mpc.py, belief_update.py
# from .irl import LearnedRewardFunction, maximum_entropy_irl, predict_trajectory, load_irl_model
# from .particle_filter import ParticleFilter
# from .mpc import game_theoretic_mpc
# from .belief_update import bayesian_update, compute_likelihood

# __all__ = [
#     "LearnedRewardFunction",
#     "maximum_entropy_irl",
#     "predict_trajectory",
#     "load_irl_model",
#     "ParticleFilter",
#     "game_theoretic_mpc",
#     "bayesian_update",
#     "compute_likelihood",
# ]
