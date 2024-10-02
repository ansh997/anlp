from pathlib import Path

scratch_path = Path('/scratch/hmnshpl/anlp_data')

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10e-4,
        "seq_len": 350,
        "d_model": 512,
        "datasource": f'{scratch_path}/ted-talks-corpus',
        "lang_src": "en",
        "lang_tgt": "fr",
        "model_folder": f"{scratch_path}/weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "/scratch/hmnshpl/anlp_data/tokenizer_{0}.json",
        "experiment_name": f"{scratch_path}/runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(scratch_path / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])