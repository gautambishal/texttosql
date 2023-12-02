import json
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from typing import Dict, List, Iterator

import os
os.environ['CURL_CA_BUNDLE'] = ''

@DatasetReader.register("sql")
class CustomSQLDatasetReader(DatasetReader):
    def __init__(
            self,
            tokenizer: Tokenizer = None,
            token_indexers: Dict[str, TokenIndexer] = None,
            lazy: bool = False,
            manual_multiprocess_sharding: bool = True
    ) -> None:
        super().__init__(manual_multiprocess_sharding=manual_multiprocess_sharding)
        self.tokenizer = tokenizer or PretrainedTransformerTokenizer(model_name="amd-nicknick/bert-base-uncased-2022_tokenizer")
        self.token_indexers = token_indexers or {"tokens": PretrainedTransformerIndexer(model_name="bert-base-uncased")}

    def apply_token_indexers(self, instance: Instance):
        for field in instance.fields.values():
            if isinstance(field, TextField):
                field.token_indexers = self.token_indexers

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path, "r") as file:
            data = json.load(file)

        for entry in data:
            question = entry["question"]
            sql_query = entry["query"]
            yield self.text_to_instance(question, sql_query)

    def text_to_instance(
            self, question: str, sql_query: str
    ) -> Instance:
        question_tokens = self.tokenizer.tokenize(question)
        question_field = TextField(question_tokens, token_indexers=self.token_indexers)
        sql_tokens = self.tokenizer.tokenize(sql_query)
        sql_query_field = TextField(sql_tokens, token_indexers=self.token_indexers)
        fields = {
            "question": question_field,
            "sql_query": sql_query_field,
        }

        return Instance(fields)
