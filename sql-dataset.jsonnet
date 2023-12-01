{
   "dataset_reader": {
      "type": "sql",
      "lazy": false
    },
   "data_loader": {
              "batch_sampler": {
                "type": "bucket",
                "batch_size": 64,
                "sorting_keys": ["source_tokens", "num_tokens"],
                "drop_last": true
              },
              "num_workers": 4,
              "max_instances_in_memory": 1000
            },
  "train_data_path": "sql_data.json",
  "model": {
    "type": "seq2seq",
    "source_embedder": {
      "type": "embedding",
      "embedding_dim": 50
    },
    "encoder": {
      "type": "gru",
      "input_size": 50,
      "hidden_size": 50,
      "num_layers": 1
    },
    "attention": {
      "type": "dot_product"
    },
    "max_decoding_steps": 20,
    "target_namespace": "sql_tokens",
    "beam_size": 5
  },
  "trainer": {
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    },
    "num_epochs": 10
  }
}