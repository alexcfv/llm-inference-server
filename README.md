# LLM Inference Server

Локальный Flask-сервер эмбеддингов для модели
[Horizon-Labs/multilingual-embedding-small](https://huggingface.co/Horizon-Labs/multilingual-embedding-small)
(141M, мультиязычная, 1024-мерные L2-нормализованные векторы, совместимые с bge-m3).
Инференс на ONNX Runtime (int8-квантованная модель, ~269 МБ, CPU).

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Запуск

```bash
python app.py
```

Конфиг по умолчанию — `config.yaml` рядом с `app.py`. Свой путь:

```bash
python app.py --config /path/to/config.yaml
```

При первом запуске модель и токенизатор скачиваются с Hugging Face Hub
в локальный кэш (~270 МБ).

### Конфигурация (`config.yaml`)

```yaml
model:
  id: Horizon-Labs/multilingual-embedding-small  # ID модели на HF Hub
  path: null            # локальный путь к model_quantized.onnx (null — скачать)
  max_length: 512       # максимум токенов на текст
  inference_batch_size: 32  # размер батча внутри ONNX

server:
  host: 0.0.0.0
  port: 8080
  threads: null         # потоки ONNX Runtime (null — все ядра)
  max_batch_size: 128   # максимум текстов в одном HTTP-запросе
```

## API

### `GET /health`

```json
{"status": "ok", "model": "Horizon-Labs/multilingual-embedding-small", "dim": 1024, "loaded": true}
```

### `POST /embeddings`

Запрос (строка или список строк):

```bash
curl -s http://127.0.0.1:8080/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input": ["Как высока Эйфелева башня?", "La tour Eiffel mesure 330 mètres."]}'
```

Ответ:

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

Векторы L2-нормализованы — косинусная сходство = скалярное произведение.
Префиксы query/passage не нужны.

```python
import numpy as np

q = np.array(resp["data"][0]["embedding"])
p = np.array(resp["data"][1]["embedding"])
similarity = q @ p
```
