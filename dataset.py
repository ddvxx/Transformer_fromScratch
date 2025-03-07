import torch
from torch.utils.data import Dataset
from typing import Any

class TranslationDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, lang_src, lang_tgt, seq_len):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src  
        self.tokenizer_tgt = tokenizer_tgt
        self.lang_src = lang_src
        self.lang_tgt = lang_tgt
        self.seq_len = seq_len
        
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx) -> Any:
        src_index_pair = self.ds[idx]
        src_text = src_index_pair['translation'][self.lang_src]
        tgt_text = src_index_pair['translation'][self.lang_tgt]

        input_tokens_encoder = self.tokenizer_src.encode(src_text).ids
        input_tokens_decoder = self.tokenizer_tgt.encode(tgt_text).ids

        # Add padding tokens to fullfil the sequence length
        padding_tokens_encoder = self.seq_len - len(input_tokens_encoder) - 2
        padding_tokens_decoder = self.seq_len - len(input_tokens_decoder) - 1

        if padding_tokens_encoder < 0 or padding_tokens_decoder < 0:
            raise ValueError('Sequence length is too long')  
        
        # Add special tokens to the tokens ids for the encoder and decoder input
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(input_tokens_encoder, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * padding_tokens_encoder)
            ],
            dim=0
        )
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(input_tokens_decoder, dtype=torch.int64),
                torch.tensor([self.pad_token] * padding_tokens_decoder, dtype=torch.int64)
            ],
            dim=0
        )

        # This is the target label we want to reach in the prediction, output of decoder, we added it EOS token
        label = torch.cat(
            [
                torch.tensor(input_tokens_decoder, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * padding_tokens_decoder, dtype=torch.int64)
            ],
            dim=0
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        # The encoder should not see PAD tokens for the self-attention layer, we mask it, as well as in the
        # decoder, we mask the PAD tokens and the future tokens
        return {
            'encoder_input': encoder_input, # (seq_len)
            'decoder_input': decoder_input, # (seq_len)
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, 1, seq_len) & (1, seq_len, seq_len)
            'label': label, # (seq_len)
            'src_text': src_text,
            'tgt_text': tgt_text
        }
    
def causal_mask(size):
    # Create a mask to prevent the decoder to see the future tokens
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0