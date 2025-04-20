# ---------------------------------------------------------------------
# Central location for all hyper‑parameters so every module can import
# the same settings without duplication.
# ---------------------------------------------------------------------

config = {
    "timesteps": 1000,
    "guidance_weight": 1.0,
    "uncond_prob": 0.1,
    "batch_size": 32,
    "hidden_dim": 128,
    "learning_rate": 4e-4,
    "ema_decay": 0.99,
}
