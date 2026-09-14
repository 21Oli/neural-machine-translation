# Neural Machine Translation (NMT)

End-to-end neural machine translation system with support for low-resource language pairs, including English ↔ Amharic.

## Project Structure

```
neural-machine-translation/
├── configs/          # YAML configuration files
├── data/             # Raw, interim, processed, and external data
├── notebooks/        # Exploratory and analysis notebooks
├── src/nmt/          # Core library source code
│   ├── data/         # Data downloading, auditing, cleaning, splitting
│   ├── tokenization/ # Tokenizer, vocabulary, serialization
│   ├── models/       # Encoder, decoder, seq2seq, attention models
│   ├── training/     # Trainer, callbacks, checkpoints
│   ├── evaluation/   # Metrics, benchmarking, error analysis
│   ├── inference/    # Translator and preprocessing
│   └── utils/        # Config, seed, logging, timing utilities
├── models/           # Saved model weights
├── artifacts/        # Tokenizers, vocabularies, metrics, predictions
├── reports/          # Figures, tables, and final report
├── tests/            # Unit tests
└── app/              # Inference API and UI
```

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

## Configuration

Copy `.env.example` to `.env` and fill in the required values.

Edit `configs/base.yaml` for model and training hyperparameters.

## Usage

### Training

```bash
python -m nmt.training.trainer --config configs/amharic.yaml
```

### Inference

```bash
python -m nmt.inference.translator --text "Hello world"
```

### App

```bash
python app/app.py
```

## Notebooks

| Notebook | Description |
|---|---|
| `01_dataset_audit` | Dataset statistics and quality audit |
| `02_preprocessing_analysis` | Preprocessing pipeline analysis |
| `03_baseline_seq2seq` | Baseline seq2seq model training |
| `04_attention_seq2seq` | Attention-based model training |
| `05_error_attention_analysis` | Error and attention visualization |

## License

See [LICENSE](LICENSE).
