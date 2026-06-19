"""Generative augmentation framework for the MMS dataset.

study  -> profile_attacks.py     (learn field/frequency/timing distributions)
model  -> attack_model.py        (conditional-empirical + Markov sampler)
reproduce -> generate_attacks.py (extend the corpus to a target attack ratio)
validate -> validate_augmentation.py (synthetic vs real fidelity report)

Balanced corpus synthesis lives in mms_dataset_augmentor.py. It profiles both
the normal MMS baseline and the attack bursts, then generates exact requested
normal/attack counts while preserving protocol groups such as WRITE control
sequences and GET_NAME_LIST request/response pairs.
"""
