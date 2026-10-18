import os

from flask import Flask, jsonify, request

from embedding_server.model import EmbeddingModel

MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "128"))


def error(message: str, status: int):
    return jsonify({"error": message}), status


def create_app(model: EmbeddingModel) -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "model": model.model_id,
                "dim": model.dim,
                "loaded": True,
            }
        )

    @app.post("/embeddings")
    def embeddings():
        payload = request.get_json(silent=True)
        if payload is None:
            return error("Request body must be valid JSON", 400)

        inp = payload.get("input")
        if isinstance(inp, str):
            texts = [inp]
        elif isinstance(inp, list) and all(isinstance(t, str) for t in inp):
            texts = inp
        else:
            return error('"input" must be a string or a list of strings', 400)

        if not texts:
            return error('"input" must not be empty', 400)
        if len(texts) > MAX_BATCH_SIZE:
            return error(
                f"Too many inputs: {len(texts)} > {MAX_BATCH_SIZE}", 400
            )

        vectors = model.encode(texts)
        return jsonify(
            {
                "data": [
                    {"index": i, "embedding": vec.tolist()}
                    for i, vec in enumerate(vectors)
                ],
                "dim": model.dim,
                "model": model.model_id,
            }
        )

    return app
