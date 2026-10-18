# LLM Inference Server

Локальный Flask-сервер эмбеддингов для модели
[Horizon-Labs/multilingual-embedding-small](https://huggingface.co/Horizon-Labs/multilingual-embedding-small)
(141M, мультиязычная, 1024-мерные L2-нормализованные векторы, совместимые с bge-m3).
Инференс на ONNX Runtime (int8-квантованная модель, ~269 МБ, CPU).

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Запуск

```bash
python app.py
```

При первом запуске модель и токенизатор скачиваются с Hugging Face Hub
в локальный кэш (~270 МБ).

### Переменные окружения

| Переменная | По умолчанию | Описание |
|---|---|---|
| `MODEL_ID` | `Horizon-Labs/multilingual-embedding-small` | ID модели на HF Hub |
| `MODEL_PATH` | — | Локальный путь к `model_quantized.onnx` (пропускает скачивание onnx-файла) |
| `HOST` | `127.0.0.1` | Адрес |
| `PORT` | `8080` | Порт |
| `THREADS` | все ядра | Число потоков ONNX Runtime |
| `INFERENCE_BATCH_SIZE` | `32` | Размер батча внутри ONNX |
| `MAX_BATCH_SIZE` | `128` | Максимум текстов в одном запросе |

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
