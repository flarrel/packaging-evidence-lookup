"""Commands for local packaging research."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path(".work"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("medical", help="prepare the PneumoniaMNIST image benchmark")
    commands.add_parser("prepare", help="build the packaging corpus")
    commands.add_parser("models", help="fetch pinned public CPU models")
    commands.add_parser("train", help="fit and save the two rankers")
    evaluate = commands.add_parser("evaluate", help="run the packaging benchmark")
    evaluate.add_argument("--split", choices=("validation", "test"), default="test")
    ask = commands.add_parser("ask", help="look up package evidence")
    ask.add_argument("question")
    ask.add_argument("--image", type=Path)
    serve = commands.add_parser("serve", help="start the loopback API")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--host", choices=("127.0.0.1", "0.0.0.0"), default="127.0.0.1")
    args = parser.parse_args()
    if args.command == "medical":
        from .medical import run

        result = run(args.work)
    elif args.command == "prepare":
        from .corpus import prepare

        result = prepare(args.work)
    elif args.command == "models":
        from .models import fetch

        fetch(args.work)
        result = {"models": "ready"}
    elif args.command == "serve":
        import uvicorn

        from .api import create_app

        uvicorn.run(create_app(args.work), host=args.host, port=args.port)
        return
    else:
        from .retrieval import Index

        index = Index(args.work)
        try:
            if args.command == "train":
                from .ranker import train

                result = train(index)
            elif args.command == "evaluate":
                from .evaluation import evaluate

                result = evaluate(index, args.split)
            else:
                from .answering import Lookup
                from .images import load_image

                image = load_image(args.image) if args.image else None
                result = Lookup(index).ask(args.question, image)
        finally:
            index.close()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
