from pathlib import Path


def log_print(s, log_file, also_print=True):
    if also_print:
        print(s)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'a') as f:
            f.write(s + "\n")