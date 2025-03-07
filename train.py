
import torchmetrics
import torch
import torch.nn as nn
import warnings
import os

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from model import Transformer, build_transformer
from dataset import TranslationDataset, causal_mask
from config import get_config, get_file_path_for_weights, lastest_weights_path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, random_split

def greedy_decode(model: Transformer, source: torch.Tensor, source_mask: torch.Tensor, tokenizer_tgt: Tokenizer, max_len: int, device: torch.device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Once we have the encoder output, we use the same on every iteration
    encoder_output = model.encode(source, source_mask)

    # As we want to make validation, we have to start with the SOS token into the decoder input
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Calculate the output of the decoder given the input and the encoder output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get the next token with the highest probability, and add it to the decoder input, we are using greedy decoding
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        # If the next wort is the EOS token, we stop the loop, we have finished the sentence
        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def run_validation(model: Transformer, validation_ds, tokenizer_src, tokenizer_tgt: Tokenizer, max_len, device, print_msg, global_step, writer: SummaryWriter, num_examples: int):
    model.eval()
    count = 0
    source_texts = []
    expected = []
    predicted = []

    try:
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80

    # Disable the gradient computation, we are only predicting
    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (batch, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (batch, 1, 1, seq_len)

            assert encoder_input.size(0) == 1, "Batch size for validation should be 1"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]

            # We transform the tensor to text moving the tensor to the CPU and using the tokenizer
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            print_msg('-' * console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-' * console_width)
                break
    
    if writer:
        # Evaluate the character error rate
        # Calculate the char error rate 
        metric = torchmetrics.CharErrorRate()
        char_error = metric(predicted, expected)
        writer.add_scalar('validation char_error', char_error, global_step)
        writer.flush()

        # Calculate the word error rate
        metric = torchmetrics.WordErrorRate()
        word_error = metric(predicted, expected)
        writer.add_scalar('validation word_error', word_error, global_step)
        writer.flush()

        # Calculate the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()

def get_all_sentences(ds, lang):
    # We create a generator to avoid loading all the dataset in memory
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))

    # If we don't have the tokenizer, build it adding the special tokens
    # We are using HuggingFace tokenizers
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    
    tokenizer_src = get_or_build_tokenizer(config, ds, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds, config['lang_tgt'])

    # Split the dataset into training and validation
    train_ds_size = int(0.9 * len(ds))
    val_ds_size = len(ds) - train_ds_size
    train_ds, val_ds = random_split(ds, [train_ds_size, val_ds_size])

    # Create the dataset with the TranslationDataset class, which will add the special tokens
    train_ds = TranslationDataset(train_ds, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = TranslationDataset(val_ds, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence to specify the model seq_len
    max_len_src = 0
    max_len_tgt = 0

    for item in ds:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source : {max_len_src}')
    print(f'Max length of target : {max_len_tgt}')
    
    # Create the dataloaders for the dataset iteration
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, len_vocab_src, len_vocab_tgt):
    model = build_transformer(len_vocab_src, len_vocab_tgt, config["seq_len"], config['seq_len'], config['d_model'],
                              config['num_encoder_decoder_layers'], config['num_heads'], config['dropout'], config['d_ff'])
    return model

def train_model(config):
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    device = torch.device(device)

    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)
    
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    # Tensorboard for experiment tracking
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], eps=1e-9)

    # Check if we have a model to preload, if so (preload == 'latest' in the config file) we load the latest model
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = (lastest_weights_path(config) 
                      if preload == 'latest' 
                      else get_file_path_for_weights(config, preload) 
                      if preload else None)
    
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('Any model is saved, starting new training')

    # Loss function, we ignore the padding token
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Running Epoch {epoch:02d}")

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (batch, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (batch, 1, seq_len, seq_len)

            # Feed the tensors to the model
            encoder_output = model.encode(encoder_input, encoder_mask) # (batch, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch, seq_len, d_model)
            proj_output = model.project(decoder_output) # (batch, seq_len, vocab_size)

            # Compare the decoder output with the target label
            label = batch['label'].to(device) # (batch, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            loss.backward()

            # With the optimizer, update the weigths
            optimizer.step()

            # Zero the gradients for the next iteration
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # We have to run the validation at the end of each epoch to see the precision of the model visually
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer, config['validation_examples'])

        # Save the model at the end of every epoch
        model_filename = get_file_path_for_weights(config, f"{epoch:02d}")

        # Save the model in the weights folder for an specific epoch, so we can have checkpoints
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)