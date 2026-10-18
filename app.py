import argparse

from embedding_server.api import create_app
from embedding_server.config import load_config
from embedding_server.model import EmbeddingModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="config.yaml", help="Path to YAML config"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = EmbeddingModel(
        model_id=cfg.model.id,
        model_path=cfg.model.path,
        max_length=cfg.model.max_length,
        batch_size=cfg.model.inference_batch_size,
        num_threads=cfg.server.threads,
    )
    app = create_app(model, max_batch_size=cfg.server.max_batch_size)
    app.run(host=cfg.server.host, port=cfg.server.port)


if __name__ == "__main__":
    main()
