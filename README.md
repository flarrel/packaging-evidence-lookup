# Packaging Evidence Lookup

Look up packaging information in a collection of U.S. drug labels from DailyMed. Give the tool an exact product name or package code, or attach a label image, then ask for the product name, package codes, or packaging description. Each answer cites the matching DailyMed label. The tool asks for clarification when it cannot identify a single product and declines questions about other fields.

The command-line interface and local HTTP API run on CPU. The image benchmark uses cropped reference artwork with adjusted brightness. Performance on independently taken photos remains unverified.

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
```

The benchmark measures how much a label image helps identify the correct product. See [RESULTS.md](RESULTS.md) for the setup, results, and failure cases.

Sources include [DailyMed](https://dailymed.nlm.nih.gov/dailymed/about-dailymed.cfm), [CLIP](https://huggingface.co/openai/clip-vit-base-patch32), and [FLAN-T5](https://huggingface.co/google/flan-t5-small). The data snapshot records public label IDs and factual fields. Package artwork remains in local storage.
