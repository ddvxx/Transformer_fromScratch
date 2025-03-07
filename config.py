from pathlib import Path

def get_config():
    return {
        'batch_size': 8,
        'epochs': 30,
        'learning_rate': 0.001,
        'seq_len': 800,
        'd_model': 512,
        'num_heads': 8,
        'num_encoder_decoder_layers': 6,
        'dropout': 0.1,
        'd_ff': 2048,
        "datasource": 'opus_books',
        'lang_src': 'en',
        'lang_tgt': 'es',
        'preload': "latest",
        'tokenizer_file': 'tokenizer_{0}.json',
        'experiment_name': 'runs/transformer',
        'model_folder': 'weights',
        'model_name': 'transformer_',
        'validation_examples': 3
    }

# Get the file where the weights on each epoch will be saved
def get_file_path_for_weights(config, epoch):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_name']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Get the latest weights file in the folder
def lastest_weights_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_name']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])

    