# Packaging evidence experiment

## Question

Does a package label image improve retrieval when a question specifies the requested field but omits the product name or code?

## Data and method

The pinned DailyMed snapshot contains 72 records from 36 product families. Training uses 44 records from 22 families. Validation and test each use 14 records from seven different families. Ingestion excluded 16 unusable or duplicate source entries. The snapshot records label versions, image URLs, and SHA-256 hashes. A record is eligible when its single-product label contains package codes, packaging text, and an image referenced from the principal display panel.

The benchmark searches the same 72-record collection for every approach. The rankers train on 5,676 positive and negative pairs within the training families. Validation loss selects a checkpoint. Held-out queries target only the test families. Each test record has one exact text query, two questions that depend on an image, one unsupported query, and one query with insufficient identifying information. The benchmark uses source artwork with deterministic cropping and brightness changes for image queries.

BM25 and CLIP text retrieval are the text baselines. Reciprocal rank fusion follows [Cormack, Clarke, and Büttcher](https://cormack.uwaterloo.ca/cormack/cormacksigir09-rrf.pdf), using the sum of one over 60 plus a candidate's rank in each input list. A PyTorch linear ranker and a small nonlinear ranker each combine lexical, text, and image scores. A text-only ablation removes the image before ranking. The local service asks for clarification when the visual match is weak or ambiguous. FLAN-T5 proposes a response, which must match the cited field. The service returns that field directly when the generated response differs.

## Held-out results

The macOS CPU run included 42 answerable queries and 28 queries that should abstain or ask for clarification.

- BM25 ranked the correct record first for 16 of 42 queries. CLIP text retrieval did so for five. Reciprocal rank fusion did so for 12.
- The nonlinear ranker ranked the correct record first for 14 of 42 queries without the image and 42 of 42 with it. The simpler linear ranker also reached 42 of 42 with the image.
- The service returned the expected cited record for all 42 answerable questions. Thirteen were verified language-model outputs, and 29 returned the field directly. All 14 unsupported questions produced an abstention. All 14 ambiguous questions requested clarification.
- Observed end-to-end latency on this machine was 32 ms at the median and 128 ms at the 95th percentile. These figures include early exits for clarification and abstention.

A clean Linux container independently prepared the pinned snapshot, trained both rankers, and repeated the test. The nonlinear ranker's top result matched for all 42 answerable questions. Serving returned the right citation for 40 of those and requested clarification on two queries for one label. Both runs returned all expected abstentions and clarification requests for the 28 queries intentionally missing support or identifying information.

## Failure analysis

Text scores could not identify a specific package when the question named only a field. Unweighted fusion also promoted competing text results for some image-dependent queries. On the validation split, the service requested clarification on two queries whose correct product still ranked first. Both validation misses came from the same Orajel label, with a CLIP similarity gap of 0.0063 between the two closest records. Reducing the required gap from 0.025 to 0.015 left all 28 image-dependent validation decisions unchanged, so the service kept the original value. The Linux rerun reproduced this kind of behavior on a different label. Ranking a label first and having enough evidence to return it are different checks.

The results apply to a small collection of reference artwork and scripted questions. The test applies controlled changes to indexed artwork. A different design, independently captured photo, glare, or occlusion can produce different results. The benchmark provides evidence for the image input on these queries. It cannot establish an advantage for the nonlinear model over the linear model.

## Medical image processing

The medical command processed all 5,856 [PneumoniaMNIST images](https://medmnist.com/v2) using the supplied splits, with 4,708 training, 524 validation, and 624 test images. It verified dimensions and labels and produced normalized 28×28 grayscale arrays, a sample grid, and counts per split. The summary, grid, and arrays were byte-identical on a rerun. The [official archive](https://zenodo.org/records/10519652) is checksum-verified before processing.
