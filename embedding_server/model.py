import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

DEFAULT_MODEL_ID = "Horizon-Labs/multilingual-embedding-small"
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 32


class EmbeddingModel:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        model_path: str | None = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_threads: int | None = None,
    ) -> None:
        self.model_id = model_id
        self.max_length = max_length
        self.batch_size = batch_size

        if model_path:
            onnx_path = Path(model_path)
            if not onnx_path.is_file():
                raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        else:
            onnx_path = Path(
                hf_hub_download(repo_id=model_id, filename="onnx/model_quantized.onnx")
            )

        tokenizer_path = hf_hub_download(repo_id=model_id, filename="tokenizer.json")
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_truncation(max_length=max_length)
        self._tokenizer.enable_padding(
            pad_id=self._tokenizer.token_to_id("[PAD]") or 0,
            pad_token="[PAD]",
        )

        sess_options = ort.SessionOptions()
        if num_threads:
            sess_options.intra_op_num_threads = num_threads
        self._session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {inp.name for inp in self._session.get_inputs()}
        self._lock = threading.Lock()

    @property
    def dim(self) -> int:
        return int(self._session.get_outputs()[0].shape[-1])

    def encode(self, texts: list[str]) -> np.ndarray:
        embeddings: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            embeddings.append(self._encode_batch(batch))
        return np.concatenate(embeddings, axis=0)

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        encodings = self._tokenizer.encode_batch(texts)
        feed = {
            "input_ids": np.array(
                [e.ids for e in encodings], dtype=np.int64
            ),
            "attention_mask": np.array(
                [e.attention_mask for e in encodings], dtype=np.int64
            ),
        }
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.array(
                [e.type_ids for e in encodings], dtype=np.int64
            )
        with self._lock:
            (sentence_embedding,) = self._session.run(
                ["sentence_embedding"], feed
            )
        return sentence_embedding
