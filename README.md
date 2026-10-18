# LLM Inference Server

Embedding server for my [Profiler](https://github.com/alexcfv/profiler) project —
it serves text embeddings that Profiler consumes over HTTP.

Runs [Horizon-Labs/multilingual-embedding-small](https://huggingface.co/Horizon-Labs/multilingual-embedding-small)
(141M params, multilingual, 1024-dimensional L2-normalized vectors, bge-m3 compatible)
on ONNX Runtime (int8 quantized model, ~269 MB, CPU).

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Running

```bash
python app.py
```

Default config is `config.yaml` next to `app.py`. Custom path:

```bash
python app.py --config /path/to/config.yaml
```

On first launch the model and tokenizer are downloaded from Hugging Face Hub
to the local cache (~270 MB).

### Configuration (`config.yaml`)

```yaml
model:
  id: Horizon-Labs/multilingual-embedding-small  # HF Hub model ID
  path: null            # local path to model_quantized.onnx (null — download)
  max_length: 512       # max tokens per text
  inference_batch_size: 32  # batch size inside ONNX

server:
  host: 0.0.0.0
  port: 8080
  threads: null         # threads for ONNX Runtime & waitress (null — 4)
  max_batch_size: 128   # max texts per HTTP request
```

## Run as a systemd service

```bash
./deploy/setup.sh
```

The script creates a venv, installs dependencies, generates a systemd unit
(paths are taken from the repo location), enables and starts the service,
then waits for `/health` to respond.

```bash
systemctl status embedding-server   # status
sudo systemctl restart embedding-server  # restart after config change
journalctl -u embedding-server -f   # logs
```

## API

### `GET /health`

```json
{"status": "ok", "model": "Horizon-Labs/multilingual-embedding-small", "dim": 1024, "loaded": true}
```

### `POST /embeddings`

Request (a string or a list of strings):

```bash
curl -s http://127.0.0.1:8080/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input": ["How tall is the Eiffel Tower?", "La tour Eiffel mesure 330 mètres."]}'
```

Response:

```json
{
  "data": [
    {"index": 0, "embedding": [0.0123, ...]},
    {"index": 1, "embedding": [-0.0456, ...]}
  ],
  "dim": 1024,
  "model": "Horizon-Labs/multilingual-embedding-small"
}
```

Vectors are L2-normalized — cosine similarity equals dot product.
No query/passage prefixes are needed.

```python
import numpy as np

q = np.array(resp["data"][0]["embedding"])
p = np.array(resp["data"][1]["embedding"])
similarity = q @ p
```
