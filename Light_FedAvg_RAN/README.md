# Lightweight Federated Learning-based Multi-center Pulmonary Ventilation Assessment by Electrical Impedance Tomography Image Sequence

Core implementation accompanying the paper **“Lightweight Federated Learning-based Multi-center Pulmonary Ventilation Assessment by Electrical Impedance Tomography Image Sequence”**.

The code provides three training and evaluation pipelines:

- **RAN:** centralized training of a residual attention network.
- **FedAvg_RAN:** sample-size-weighted federated averaging with the RAN backbone.
- **Light_FedAvg_RAN:** federated training with sparse model updates, error feedback, optional block freezing, and pruning followed by fine-tuning.

These entry points operate on tabular ventilation features extracted from EIT image sequences. Image reconstruction and feature extraction from raw image sequences are outside the scope of these 12 scripts. Federated clients are simulated within a single Python process by partitioning a training dataset.

## Core repository structure

```text
fed_light_ran/
├── README.md
├── requirements.txt
├── .gitignore
├── main_ran.py
├── main_fed.py
├── main_light_mix_fed.py
├── models/
│   ├── __init__.py
│   ├── Nets.py
│   ├── Fed.py
│   ├── Update.py
│   └── Update_light_mix.py
└── utils/
    ├── __init__.py
    ├── options.py
    └── sampling.py
```

## Script descriptions

| File | Purpose |
| --- | --- |
| `main_ran.py` | Loads and standardizes tabular data, trains the centralized RAN using AdamW, saves the checkpoint with the highest test accuracy, and generates training curves, a confusion matrix, and a classification report. |
| `main_fed.py` | Coordinates standard federated training: samples participating clients, runs local training, aggregates model weights by client sample count, and evaluates the global model. Also produces client population distribution plots and pairwise KL-divergence statistics. |
| `main_light_mix_fed.py` | Coordinates lightweight federated training and sample-size-weighted aggregation of model deltas. Maintains client error-feedback residuals, supports global pruning followed by mask-constrained federated fine-tuning, and reports accuracy, estimated communication volume, and serialized model size. Includes optional CPU dynamic quantization for evaluation. |
| `models/__init__.py` | Marks `models` as a Python package. |
| `models/Nets.py` | Defines `RANTabular` and its building blocks, including residual MLP blocks, channel attention, and attention residual blocks. Also contains fusion and quantization-related model utilities. |
| `models/Fed.py` | Implements `FedAvg_Weighted`, which combines client model state dictionaries according to their training sample counts. |
| `models/Update.py` | Defines the tabular PyTorch dataset and the standard client-side local training and evaluation routines. |
| `models/Update_light_mix.py` | Defines lightweight local training, optional freezing of initial attention residual blocks, model-delta computation, Top-k sparsification with error feedback, and gradient masking during pruning fine-tuning. Also contains a pruning helper. |
| `utils/__init__.py` | Marks `utils` as a Python package. |
| `utils/options.py` | Defines command-line arguments for dataset paths, feature columns, model architecture, training, client partitioning, and lightweight optimization. The current parser also contains arguments for experiments outside this core release. |
| `utils/sampling.py` | Loads federated data, encodes labels, creates a stratified train/test split, fits standardization on the training set, and partitions training samples into IID or Dirichlet Non-IID clients. Preserves population subtype metadata for client statistics. |
