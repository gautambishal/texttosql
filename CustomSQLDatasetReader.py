import json
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
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
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer or PretrainedTransformerTokenizer(model_name="amd-nicknick/bert-base-uncased-2022_tokenizer")
        self.token_indexers = token_indexers or {"tokens": PretrainedTransformerIndexer(model_name="bert-base-uncased")}

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path, "r") as file:
            for line in file:
                data = json.loads(line)
                question = data["question"]
                sql_query = data["query"]
                yield self.text_to_instance(question, sql_query)

    def text_to_instance(
            self, question: str, sql_query: str
    ) -> Instance:
        question_tokens = self.tokenizer.tokenize(question)
        question_field = TextField(question_tokens, self.token_indexers)
        # Assuming your SQL queries are sequences of tokens
        sql_tokens = self.tokenizer.tokenize(sql_query)
        sql_query_field = TextField(sql_tokens, self.token_indexers)
        fields = {
            "question": question_field,
            "sql_query": sql_query_field,
        }

        return Instance(fields)
