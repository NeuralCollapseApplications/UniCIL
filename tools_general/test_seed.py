import numpy as np


if __name__ == '__main__':
    for seed in range(2020, 2040):
        SEED_INC = seed + (2**31-1)
        rng1 = np.random.default_rng(SEED_INC)
        session_types = rng1.choice(['plain', 'lt', 'fs'], 10, replace=True)
        print(f"{seed}: {session_types}")
    