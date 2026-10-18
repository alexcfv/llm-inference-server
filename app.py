import os

from embedding_server.api import create_app
from embedding_server.model import DEFAULT_MODEL_ID, EmbeddingModel


def main() -> None:
    model = EmbeddingModel(
        model_id=os.environ.get("MODEL_ID", DEFAULT_MODEL_ID),
        model_path=os.environ.get("MODEL_PATH"),
        batch_size=int(os.environ.get("INFERENCE_BATCH_SIZE", "32")),
        num_threads=(
            int(os.environ["THREADS"]) if os.environ.get("THREADS") else None
        ),
    )
    app = create_app(model)
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8080")),
    )


if __name__ == "__main__":
    main()
