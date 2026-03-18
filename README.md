# Decontamination

An n-gram based decontamination tool for LLM training datasets.

--------------------

# Installation

```bash
uv pip install https://github.com/ali-elganzory/decontamination.git
```

# Usage

All commands can be configured using either:
- command arguments. Check
  - `es --help`
  - `ds --help`
  - `ds remove --help`
- a YAML config file with the `--config <config_file>` argument. Check
  - `configs/es/build.yaml`
  - `configs/es/prepare.yaml`
  - `configs/es/run.yaml`
  - `configs/ds/index.yaml`
  - `configs/ds/search.yaml`
  - `configs/ds/remove.yaml`

### 1. Run `elasticsearch` container

```bash
es build         # Build an Elasticsearch apptainer image
es prepare       # Prepare the image mounts and single-node configuration
es run           # Run the Elasticsearch container
```

Elasticsearch is used as a search engine to efficiently index the training datasets and perform n-gram search for overlap with evaluation benchmarks.  

### 2. Index datasets into Elasticsearch

```bash
ds index
```

The datasets to be indexed are specified in the YAML config file (check `configs/datasets.yaml`).

```yaml
- ...
  path: <dataset_path>
  subset: <subset>
  split: <split>
  ...
```

### 3. Find contaminated samples

```bash
ds search
```

The benchmarks to be searched for contamination are specified in the YAML config file (check `configs/benchmarks.yaml`).

```yaml
- ...
  path: <benchmark_path>
  subset: <subset>
  split: <split>
  ...
```

The search results are saved to `outputs/results.csv`.

### 4. Remove contaminated samples

```bash
ds remove --huggingface-id <huggingface-id>
```

The decontaminated dataset is saved to 
- `outputs/<dataset_name>-decontaminated/` locally, and
- `<huggingface-id>/<dataset_name>-decontaminated/` on Hugging Face (if `--huggingface-id` is provided).

--------------------

### Command abbreviations

- `es` -> `elasticsearch`
- `ds` -> `datasets`

--------------------

# Acknowledgements

The decontamination scripts in this project are adapted from [allenai/open-instruct](https://github.com/allenai/open-instruct), AllenAI's post-training codebase. See their [Contamination checks](https://github.com/allenai/open-instruct#contamination-checks) section for the original implementation.

--------------------

# License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
