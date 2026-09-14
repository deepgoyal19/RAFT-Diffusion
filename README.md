# RAFT-Diffusion

**Reward-ranked fine-tuning for text-to-image diffusion models.**

This repository contains diffusion-model research code developed in 2023 under
the original name **LMFlow-diffusion**. It implements a generate, rank, select,
and fine-tune workflow for Stable Diffusion, with both full UNet fine-tuning and
LoRA-based adaptation paths.

Related paper: [RAFT: Reward rAnked FineTuning for Generative Foundation Model
Alignment](https://arxiv.org/abs/2304.06767), published in *Transactions on Machine
Learning Research* (TMLR).

This is a historical research implementation, not a newly benchmarked release or
a claim that every result in the paper can be reproduced from this snapshot.

## How it works

Each RAFT round in this implementation:

1. **Generates candidates:** sample multiple images for each input text prompt.
2. **Scores candidates:** evaluate images using an aesthetic, CLIP, or PickScore
   scoring backend.
3. **Selects training pairs:** keep the highest-scoring image for each prompt,
   then select the global top-`k` image/prompt pairs for the round.
4. **Fine-tunes the diffusion model:** train on those selected pairs using the
   diffusion denoising objective.
5. **Repeats:** generate new candidates with the updated model in the next round.

The reward is used to **select training data**. The code does not backpropagate
through the reward model or optimize a PPO-style policy objective. Its training
loss is mean-squared error against the sampled noise or velocity target,
depending on the scheduler's prediction type; optional SNR weighting is exposed.

With `use_lora=True`, the base UNet is frozen and LoRA attention processors are
trained. Otherwise, the UNet is fine-tuned directly. The VAE and text encoder are
frozen in both paths.

## Repository map

| Path | Purpose |
| --- | --- |
| [`examples/raft.py`](examples/raft.py) | RAFT command-line entry point |
| [`examples/finetuner.py`](examples/finetuner.py) | Supervised image/text fine-tuning entry point |
| [`examples/inferencer.py`](examples/inferencer.py) | Image generation entry point |
| [`src/lmflow_aigen/args.py`](src/lmflow_aigen/args.py) | Model, data, training, inference, and RAFT arguments |
| [`src/lmflow_aigen/pipeline/diffusion_finetuner.py`](src/lmflow_aigen/pipeline/diffusion_finetuner.py) | Candidate selection and diffusion training loops |
| [`src/lmflow_aigen/models/diffusion_model.py`](src/lmflow_aigen/models/diffusion_model.py) | Model loading, LoRA, scoring, and checkpoint saving |
| [`src/lmflow_aigen/datasets/dataset.py`](src/lmflow_aigen/datasets/dataset.py) | Prompt loading and selected image/text dataset preparation |
| [`scripts/`](scripts/) | Original experiment launch scripts |
| [`configs/`](configs/) | Original Accelerate configuration |

The Python import namespace remains `lmflow_aigen`; renaming the public
repository does not rename the package.

## Environment and setup

The original code targets CUDA-based PyTorch and the Hugging Face Diffusers,
Transformers, Datasets, and Accelerate stack. Package metadata lists Python 3.9
and 3.10, and `requirements.txt` specifies PyTorch 2.0 or later.

**An exact working environment is not pinned in this repository.** The code uses
legacy Diffusers APIs such as `LoRAAttnProcessor` and `AttnProcsLayers`. Installing
the newest dependency versions is not a verified setup procedure.

To obtain the source:

```bash
git clone https://github.com/deepgoyal19/RAFT-Diffusion.git
cd RAFT-Diffusion
```

Prepare an isolated environment with mutually compatible versions of the
dependencies before installing the package. In addition to
[`requirements.txt`](requirements.txt), the source imports `datasets`,
`torchvision`, and the OpenAI `clip` package. The default logging backend needs
TensorBoard; W&B is optional. xFormers is needed for the paths that enable it.

Once those dependencies are available:

```bash
python -m pip install --no-deps -e .
accelerate config
python examples/raft.py --help
```

These are setup and argument-discovery instructions, not a tested end-to-end
reproduction recipe. No training or model downloads were run for this README
update.

## Data and configuration

### RAFT prompt data

For a local prompt dataset, point `--train_data_dir` at a directory containing a
text file with one prompt per line, and keep `--overrode_init_dataset True`:

```text
data/prompts/train.txt
```

Example file contents:

```text
A watercolor landscape with mountains and a lake.
A photograph of a red bicycle beside a brick wall.
```

Alternatively, `--dataset_name` uses the Hugging Face dataset loader. The RAFT
generation loop expects a `train` split with a `text` column.

For ordinary supervised fine-tuning, the separate entry point uses image/caption
pairs; see [`scripts/run_finetuner.sh`](scripts/run_finetuner.sh) and set
`--overrode_init_dataset False`.

### Key RAFT arguments

| Argument | Role |
| --- | --- |
| `--pretrained_model_name_or_path` | Stable Diffusion model identifier or local model directory |
| `--num_images_per_prompt` | Candidate images generated per prompt |
| `--inference_batch_size` | Prompts processed together during candidate generation |
| `--score_model` | `aesthetic`, `clip`, or `pick` scoring path |
| `--clip_model_pretrained_or_path` | Model identifier passed to the chosen scoring backend |
| `--topk` | Number of selected image/prompt pairs retained globally per round, not per prompt |
| `--epochs` | Number of RAFT generation/selection/fine-tuning rounds |
| `--max_train_steps` | Per-round training-step budget used to advance the cumulative step limit |
| `--train_batch_size` | Fine-tuning batch size per device |
| `--use_lora` / `--rank` | Enable LoRA and set its rank |
| `--output_dir` | Checkpoint and generated-image output directory |

[`scripts/run_raft.sh`](scripts/run_raft.sh) records the original experiment
arguments. **Do not launch it unchanged:** it contains machine-specific absolute
paths, and the bundled Accelerate configuration selects GPU `7`. Choose your own
model/data/output locations, device, and memory budget.

The example programs also accept a single JSON configuration file through
`HfArgumentParser`. Argument names, including the historical `overrode_*`
spellings, must match [`args.py`](src/lmflow_aigen/args.py).

## Reproducibility notes

This snapshot includes known implementation limitations, not just dependency
compatibility questions:

- In `raft_finetune`, the generation pipeline is deleted inside the prompt-batch
  loop but constructed outside that loop. Multiple prompt batches need that
  lifecycle issue addressed before use.
- The returned `finetuned_model_pipeline` is only assigned when
  `save_finetune_images` is enabled. The disabled path needs correction before
  relying on the return value.
- The standalone inferencer loads a full Stable Diffusion pipeline; its
  `use_lora` flag alone does not load a separately saved LoRA checkpoint.
- Some image-saving paths remove existing generated-image directories. Use a
  fresh output directory and retain any outputs you want to keep.

No benchmark tables, trained model weights, or locked reproduction environment
are included in this release. The paper's results should not be presented as
measurements newly obtained from this repository.

## Attribution and history

The preserved development history includes contributions from **Deepanshu
Goyal** and **Hanze Dong**. Original source headers credit the Statistics and
Machine Learning Research Group at HKUST, and package metadata credits the
LMFlow Team.

This standalone public release was made with the original repository owner's
permission. Original commit authors and timestamps were retained. A historical
credential was removed before publication, which changed affected commit hashes
and their descendants. Documentation updates have their actual new dates.

No repository-wide license file is included in this snapshot. Existing copyright
notices are retained; contact the maintainers to clarify reuse terms.

## Citation

Please cite the RAFT paper when referring to the method:

```bibtex
@misc{dong2023raft,
  title = {RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment},
  author = {Hanze Dong and Wei Xiong and Deepanshu Goyal and Yihan Zhang and
            Winnie Chow and Rui Pan and Shizhe Diao and Jipeng Zhang and
            Kashun Shum and Tong Zhang},
  year = {2023},
  eprint = {2304.06767},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2304.06767}
}
```
