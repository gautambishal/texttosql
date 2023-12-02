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
      "max_instances_in_memory": 1000
  },
  "train_data_path": "sql_data.json",
  "model": {
      "type": "seq2seq",
      "embedder": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 50
        }
    },
    "encoder": {
      "type": "gru",
      "input_size": 50,
      "hidden_size": 50,
      "num_layers": 1
    },
    "decoder": {
      "type": "gru",
      "input_size": 50,
      "hidden_size": 50,
      "num_layers": 1,
      "attention": {
        "type": "linear",
        "tensor_1_dim": 50,
        "tensor_2_dim": 50
      }
    }
  },
  "trainer": {
      "optimizer": {
        "type": "adam",
        "parameter_groups": [
          {
            "params": ["model.*"],
            "lr": 0.001
          },
          {
            "params": ["model.encoder.*"],
            "lr": 0.0001
          }
        ]
      },
      "num_epochs": 10
    }
}