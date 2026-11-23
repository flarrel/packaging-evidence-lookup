# Packaging Evidence Lookup

Look up a product name, package code, or packaging description in a small DailyMed label collection. Submit an exact product name or package code, or attach an image of the package label. The result includes a link to the source label. Ambiguous matches prompt for clarification. Questions outside the supported fields receive an abstention.

The project is a local research prototype for packaging metadata lookup. The command-line interface and HTTP service run on CPU.

## Quick start

Use Python 3.12 and run these commands from the repository root.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m label_lookup prepare
python -m label_lookup models
python -m label_lookup train
python -m label_lookup ask 'Which package codes are listed for Advil Menstrual Pain?'
```

Setup downloads roughly 253 MB of model files plus the package artwork and Python dependencies. Data, checkpoints, and reports are stored under `.work/` by default. To place them somewhere else, pass `--work PATH` before the command. Preparation and model downloads need network access. Training, lookup, and evaluation use the local files.

Use `--image PATH` for a package photo or image. The `prepare --discover` command can build a fresh snapshot from DailyMed for a new experiment. The default `prepare` command uses the benchmark snapshot with pinned source versions and image checksums.

## Local API

```sh
python -m label_lookup serve
curl http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"Which package codes are listed for Advil Menstrual Pain?"}'
```

The optional `image_base64` field accepts a base64-encoded image. See `http://127.0.0.1:8000/docs` for the HTTP schema. Query responses show the status, citation, and steps taken. A verified model answer must match its source value. The service returns that source value directly if the model deviates.

For Docker, reuse the same inputs and checkpoints.

```sh
docker build -t packaging-evidence-local:dev .
docker run --rm --network none -v "$PWD/.work:/work" \
  packaging-evidence-local:dev --work /work evaluate
docker run --rm -p 127.0.0.1:8000:8000 -v "$PWD/.work:/work" \
  packaging-evidence-local:dev --work /work serve --host 0.0.0.0
```

Shut down the service before running another command on the same work directory. The embedded vector database opens its storage in one process at a time.

## Experiments

```sh
python -m label_lookup evaluate --split validation
python -m label_lookup evaluate
python -m unittest discover -s tests -v
python -m label_lookup medical
```

The benchmark compares lexical, dense text, fused, and learned image-text ranking. The medical command validates and normalizes the public PneumoniaMNIST sample and writes an image grid and per-split summary under `.work/medical/`. It is an image-processing benchmark. See [RESULTS.md](RESULTS.md) for the experiment protocol and results.

Sources include [DailyMed](https://dailymed.nlm.nih.gov/dailymed/about-dailymed.cfm), [the official MedMNIST release](https://zenodo.org/records/10519652), [CLIP](https://huggingface.co/openai/clip-vit-base-patch32), and [FLAN-T5](https://huggingface.co/google/flan-t5-small). The data snapshot records public label IDs and factual fields. Package artwork remains in local storage.
