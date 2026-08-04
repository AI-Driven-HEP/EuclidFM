""" Use ``python scripts/run_training.py`` for
new runs. This wrapper remains so existing commands do not break.
"""

from scripts.run_training import main


def train():
    return main()


if __name__ == "__main__":
    train()
