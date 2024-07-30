import os
import fnmatch

def get_latest_checkpoint_dir(model_dir):
    """Returns latest checkpoint within the given model dir"""
    sub_dir = os.listdir(model_dir)
    out = [pth for pth in sub_dir if fnmatch.fnmatch(pth, "checkpoint*")]
    out.sort()
    print(f"Getting path to saved checkpoint: {out[-1]}")
    checkpoint_dir = os.path.join(model_dir, out[-1])
    return checkpoint_dir