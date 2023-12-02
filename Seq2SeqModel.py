from allennlp.models import Model
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.attention import LinearAttention
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import BLEU
from allennlp.nn import util

import torch
import torch.nn as nn
import torch.nn.functional as F

@Model.register("seq2seq")
class Seq2SeqModel(Model):
    def __init__(self, vocab, embedder, encoder, decoder, max_decoding_steps=20):
        super(Seq2SeqModel, self).__init__(vocab)
        self.embedder = embedder
        self.encoder = PytorchSeq2SeqWrapper(encoder)
        self.decoder = decoder
        self.max_decoding_steps = max_decoding_steps

        # BLEU metric for evaluation
        self.bleu = BLEU(exclude_indices={vocab.get_token_index("[PAD]", "tokens")})


    def forward(self, source_tokens, target_tokens=None):
        source_mask = get_text_field_mask(source_tokens)
        target_mask = get_text_field_mask(target_tokens)

        embedded_source = self.embedder(source_tokens)
        encoded_source = self.encoder(embedded_source, source_mask)

        if target_tokens is not None:
            embedded_target = self.embedder(target_tokens)
            logits = self.decoder(encoded_source, source_mask, embedded_target, target_mask)
            output_dict = {"logits": logits}
            if target_tokens is not None:
                loss = sequence_cross_entropy_with_logits(logits, target_tokens["tokens"], target_mask)
                self.bleu(logits, target_tokens["tokens"], target_mask)
                output_dict["loss"] = loss
        else:
            output_dict = {"logits": None}

        return output_dict

    def get_metrics(self, reset: bool = False):
        metrics = {"bleu": self.bleu.get_metric(reset=reset)}
        return metrics
