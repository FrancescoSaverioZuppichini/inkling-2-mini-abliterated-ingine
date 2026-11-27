# Inkling-2-Mini abliterated: ingine

An abliterated [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) served
by **ingine**, a custom CUDA inference engine for one RTX 5090. It answers harmful test prompts instead of refusing
them, directly and after reasoning (refusal phrases in 1 of 104 and 3 of 48 answers, the 3 followed by the requested
text), and stays close to the original elsewhere (KL divergence 0.014 on harmless answers). ingine
generates 490-560 tokens/s, reads prompts at 6,200-10,900 tokens/s, takes images as well as text and serves
262,144 tokens of context through an OpenAI-compatible API.

The installer sets everything up and downloads the ready-to-run model from Hugging Face,
[satellitedown/Inkling-2-Mini-abliterated-ingine](https://huggingface.co/satellitedown/Inkling-2-Mini-abliterated-ingine)
(20.2 GB, pinned to one revision and checked file by file).

## Run

You need **Linux x86_64, an NVIDIA GPU with compute capability 12.0 and at least 32 GB of memory (RTX 5090), an
NVIDIA driver for CUDA 13 (580 or newer), a C++20 compiler (GCC 12 to 16; the menu offers to install one), and
about 24 GB of free disk** (20.2 GB of model files; the private CUDA toolkit, the build tools and the build take
about 1.2 GB, uv's download cache about 1 GB more). The RTX PRO 6000, 5000 and 4500 Blackwell cards are also
compute capability 12.0 but untested. Other GPUs are not supported: the kernels are built for `sm_120a` and decode
the routed experts with Blackwell's FP4 conversion instructions. The installer does not install or change GPU
drivers. The server is a C++/CUDA program: it needs no Python or PyTorch to run.

Install with one command. It clones this repository to `~/inkling-2-mini-abliterated-ingine` and opens its menu:

```bash
curl -fsSL https://raw.githubusercontent.com/satellitedown/inkling-2-mini-abliterated-ingine/main/get.sh | bash
```

Choose **1** to install (build tools, CUDA toolkit, build of the C++ engine and model download), then **5** to add
the `ingine` command and `ingine start` to run the server. Run the same command again any time to get back to the
menu.
To install somewhere else, set the folder:
`curl -fsSL https://raw.githubusercontent.com/satellitedown/inkling-2-mini-abliterated-ingine/main/get.sh | INGINE_DIR=/path/to/folder bash`.

Or do the same by hand:

```bash
git clone https://github.com/satellitedown/inkling-2-mini-abliterated-ingine
cd inkling-2-mini-abliterated-ingine
bash setup.sh
```

The menu:

1. **Install everything:** check the GPU, driver, tools and disk space, install CMake, Ninja and a private CUDA
   13.4 toolkit (with cuBLAS), build the C++ engine (`ingine-serve`, a few minutes) and download the model.
2. **Download / resume model:** download the pinned model files, or resume an interrupted download, and verify them.
3. **Start the server (foreground):** run `ingine serve` in this terminal; **Ctrl+C** stops it.
4. **Update:** pull this repository and rerun the install steps (see below). `bash setup.sh --update` does the same
   without the menu.
5. **Install the `ingine` command:** link `~/.local/bin/ingine` to `scripts/ingine` (make sure `~/.local/bin` is on
   your `PATH`).

Choose **0** to exit. Every step can be repeated: an interrupted install continues with option 1, an interrupted
download with option 2. `bash setup.sh --help` lists the options without installing anything.

**Updating** (`ingine update`, menu option 4 or `bash setup.sh --update`) fast-forwards this repository from
GitHub, then reruns the idempotent install steps: build tools, CUDA toolkit, an incremental rebuild of the engine,
and the model check. The model is downloaded again only when the pinned revision in
[runtime-manifest.json](runtime-manifest.json) changed; files that no longer match are moved aside to
`models/Inkling-2-Mini.previous-<time>/`, never deleted. Delete that folder once the new version works. A running
server keeps the old code until you restart it: `ingine stop && ingine start`. Installs from before the C++ engine
kept a Python environment in `.venv/` and compiled kernels in `engine/`; the update leaves them in place and
prints the command that deletes them (about 6 GB).

Everything lives inside the install folder:

| Path | Contents |
|---|---|
| `.tools/` | uv (when it was not installed already) and a small Python environment (`.tools/venv/`) with CMake, Ninja and `huggingface_hub`, used only by the installer |
| `.cuda-toolkit/` | private CUDA 13.4 toolkit (nvcc, runtime, cuBLAS) used to build ingine; the server loads its CUDA libraries from here, and the system CUDA is not used or changed |
| `native/build-install/` | the built server, `apps/serve/ingine-serve`, and its serving profile (`profile.env`, from [runtime-manifest.json](runtime-manifest.json)) |
| `models/Inkling-2-Mini/` | the model from Hugging Face: Inkling-2-Mini's files and the abliteration file (set `INGINE_MODEL_DIR` to use a download elsewhere) |
| `.run/` | pid file and server log, used only when there is no systemd user session (`ingine start` then runs the server with `nohup`) |

Outside it the installer writes one driver setting, `~/.nv/nvidia-application-profiles-rc.d/50-ingine`: an NVIDIA
application profile that lets `ingine-serve` use the GPU's top performance state (P0) instead of the lower one the
driver holds CUDA programs at. On an RTX 5090 that raises the memory clock by 1.4% and decoding speed by about
0.8%; outputs and idle power are unchanged. It needs driver 595 or newer;
`INGINE_GPU_PROFILE=0 bash setup.sh --update` removes it.

## RTX 3090 and RTX 4090

The installer targets the RTX 5090. On a 24 GB RTX 3090 (sm_86) or RTX 4090 (sm_89), build by hand with a system CUDA 13
toolkit, CMake 3.28 or newer and GCC 12 to 16, then start:

```bash
cd native && cmake --preset ampere && cmake --build --preset ampere && cd ..   # sm_86 and sm_89 in one binary
./start.sh <Inkling-2-Mini folder>      # http://127.0.0.1:8003/v1; PORT=..., MAX_CTX=... and extra flags work too
```

`start.sh` reads the GPU and starts the profile for agentic coding with 512,000 tokens of context. Both 24 GB cards hold 524,288 (`MAX_CTX=524288`); the default leaves ~150 MiB of headroom, so the GPU must not drive a desktop or run other programs. Flags given after the folder win over it. On start-up the server prints what it loaded:

```
ingine-serve · inkling-2-mini · ready in 8.3 s
  gpu       NVIDIA GeForce RTX 4090 (sm_89, 25.3 GB)
  weights   int8 dense
  context   512,000 tokens · KV k8v4 6.4 GB · prefill chunk 2048
  drafter   DSpark, threshold 0.8
  parking   16.3 GB host RAM · 2 conversations at full context
  memory    25.0 GB of 25.3 GB in use
  sampling  temperature 1 · top-p 1 · top-k 0
  images    off
  serving   http://127.0.0.1:8093/v1
```

### Profiles per GPU

| GPU | weights | KV cache | context | prefill chunk | embedding table | images |
|---|---|---|---:|---:|---|---|
| RTX 3090 24 GB | INT8 | k8v4 | 512,000 | 2048 | host RAM | off |
| RTX 4090 24 GB | INT8 | k8v4 | 512,000 | 2048 | host RAM | off |
| RTX 5090 32 GB | FP8 | FP8 | 512,000 (fits on paper; untested) | 3072 | GPU | on |

**Why these choices:**
- **INT8 weights:** the same size as FP8 (1 GB less than BF16). They are as fast or faster: +2.5 % over FP8 on the 3090, equal on the 4090. They are also closer to BF16: about 20 % lower KL; next-token agreement 97–98 % for both at 32K–128K, and 97.5 % vs 95.0 % at 230K on the 4090.
- **k8v4 KV:** 8-bit keys and 4-bit values, 6.06 bits per value. Keys go through the softmax, which amplifies their errors; values are only averaged.
- **Prefill chunk 2048:** 3072 reads long prompts 6–10 % faster but needs 0.37 GB more, which these cards spend on context.
- **Embedding table in host RAM:** 0.86 GiB; the GPU reads one row per token over PCIe.
- **No images:** the image tower takes 0.6 GB plus up to 0.85 GiB while encoding.

### Options

`ingine-serve --help` lists them all. The ones that matter:

| option | default | what it does |
|---|---|---|
| `--model-dir DIR` | required | the Inkling-2-Mini checkpoint |
| `--engine` / `--fake` | required, one of them | serve the model / serve canned answers (tests) |
| `--host`, `--port` | 127.0.0.1, 8003 | where the OpenAI-compatible API listens |
| `--max-ctx N` | 65,536 | tokens the full-attention caches hold (`start.sh`: 512,000) |
| `--kv-cache F` | int8 below compute capability 8.9 (3090), fp8 from 8.9 (4090, 5090) | full-attention cache format: `int8` (8.13 bits per value), `fp8` (8), `k8v4` (6.06), `fp4` (4.5), `int4` (4.06), `bf16` (exact, 16). `--kv-fp8` = `--kv-cache fp8` |
| `--dense F` | bf16 | dense weights (projections, wo, head): `bf16` (the checkpoint), `int8` or `fp8` (one byte plus a scale per row, quantized at load, about 1 GB less). `--dense-fp8` = `--dense fp8` |
| `--spec`, `--spec-threshold X` | off, 0.8 | DSpark speculative decoding: a drafter proposes up to 7 tokens and the model checks them all, so the output does not change |
| `--prefill-chunk N` | 3072 | prompt tokens per prefill step; each 1,024 tokens of chunk need about 0.37 GB |
| `--embed-host` | off | keep the embedding table (0.86 GiB) in pinned host RAM |
| `--no-vision` | images on | do not load the image tower (0.6 GB) and reject image requests |
| `--park-host-mb N` | auto | pinned host RAM for conversations displaced by another one; switching back restores them instead of re-reading the prompt. Auto: max(two conversations at `--max-ctx`, min(RAM / 4, 32 GiB)), at most RAM / 2. 0 turns it off |
| `--temperature`, `--top-p`, `--top-k` | from `generation_config.json` | sampling for requests that set none |
| `--image-max-tokens`, `--max-images`, `--max-image-tokens` | 4096, 16, 32768 | image limits per image and per request |
| `--max-queue N` | 4 | requests that may wait behind the running one |
| `--max-model-len N` | `--max-ctx` | the context limit the API reports and enforces |
| `--ablation FILE` | none | the abliteration direction file (`inkling-2-mini-abliterated`) |
| `--log-level L` | INFO | DEBUG, INFO, WARNING or ERROR |

### Speed and context on 24 GB

Measured on the round-5 code, one server per profile, one window at a time, with the drafter on (how the server runs).
- **Setup:** power caps 300 W (3090) and 400 W (4090), greedy decoding.
- **Prefill:** prompt tokens ÷ time to first token.
- **Decode:** 127 tokens after the first, with the prompt already cached.
- **Windows:** 4K / 32K / 224K / 512K.
- **Needle:** a code at 50 % depth, found in every cell.

**Drafter on, the speed you get when serving (tokens/s):**

| GPU · profile | max context | prefill 32K / 224K / 512K | decode 4K / 32K / 224K / 512K |
|---|---:|---|---|
| RTX 3090 · `start.sh` (INT8 + k8v4) | 524,288 | 4,972 / 3,063 / 1,959 | 209.3 / 223.3 / 177.1 / 124.2 |
| RTX 3090 · BF16 + int8 KV (its default KV) | 311,296 | 5,001 / 3,031 / — | 209.6 / 174.1 / 124.7 / — |
| RTX 4090 · `start.sh` (INT8 + k8v4) | 524,288 | 10,556 / 7,179 / 4,851 | 318.6 / 236.7 / 220.3 / 151.2 |
| RTX 4090 · BF16 + fp8 KV (its default KV) | 311,296 | 10,942 / 7,590 / — | 227.0 / 221.1 / 207.5 / — |

**What the tables show:**
- **Low-bit caches are faster at depth, not only smaller:** they read fewer bytes per token. At 224K, `start.sh` decodes 42 % faster than BF16 weights with the default cache on the 3090 (177.1 vs 124.7 tokens/s) and 6 % faster on the 4090 (220.3 vs 207.5).
- **More context:** `--kv-cache int4` (4.06 bits) holds 786,432 tokens (3090) / 770,048 (4090) with INT8 weights. At 512K it decodes 160 / 202 tokens/s with the drafter on.
- **Switching conversations:** coming back to a parked 229K-token conversation takes **0.86 s** to the first token (4090), against 31.4 s to read it again.

LLM_SERVING_BENCH

These GPUs have no thread-block clusters and no FP4 convert, and the 3090 has no FP8 convert either, so they take their own kernel paths:
- exact decoders;
- a cooperative exchange for the router and message gates;
- one-wave attention;
- a low-bit KV reader.

The `ampere-dev` preset adds the tests.

## Use

```bash
ingine start      # start the server in the background (systemd user unit inkling-ingine), wait until it is ready
ingine stop       # stop it and free the GPU
ingine status     # unit state, health, served model, GPU memory; exit code 0 ready, 1 starting, 3 stopped
ingine logs       # server log; ingine logs -f follows it
ingine serve      # run in the foreground instead; extra arguments go to the server, e.g. --max-ctx 131072
ingine update     # same as bash setup.sh --update
```

Without the symlink from option 5, run `scripts/ingine` from the install folder. Only one copy of the model fits
on a 32 GB card; stop other programs that hold a lot of GPU memory first. `ingine start` and `ingine stop` are
non-interactive, return when the server is ready or stopped, and send a desktop notification when `notify-send`
is available; if you like, bind them to keys in your desktop environment.

Settings come from the profile in [runtime-manifest.json](runtime-manifest.json) and can be overridden in
`~/.config/ingine/env` (shell assignments, read by every `ingine` command) or in the environment:

| Variable | Default | Meaning |
|---|---|---|
| `INGINE_MODEL_DIR` | `models/Inkling-2-Mini` in the install folder | model folder; point it at an existing download of `satellitedown/Inkling-2-Mini-abliterated-ingine` to reuse it |
| `INGINE_ABLATION` | `$INGINE_MODEL_DIR/inkling-2-mini-abliterated.safetensors` | abliteration file; a relative path is taken relative to the install folder; empty serves the original model (model id `inkling-2-mini`) |
| `INGINE_CTX` | 262144 | context length; lower it to leave GPU memory to other programs |
| `HOST`, `PORT` | 127.0.0.1, 8003 | bind address and port |
| `INGINE_EXTRA_ARGS` | empty | further server arguments, e.g. `--top-p 0.95` (sampling for requests that set none, also `--temperature`, `--top-k`), `--park-host-mb 0` (see below), `--spec-threshold 0.9` |
| `INGINE_START_TIMEOUT` | 300 | seconds `ingine start` waits for the server to report ready |

```bash
# ~/.config/ingine/env
INGINE_MODEL_DIR=/data/models/Inkling-2-Mini
INGINE_CTX=131072
```

`INGINE_MODEL_DIR` must hold the pinned revision; this checks it without downloading:
`.tools/venv/bin/python scripts/download_model.py --model-dir "$INGINE_MODEL_DIR" --verify-only`.

## Connect an application

Use the OpenAI-compatible API at **`http://127.0.0.1:8003/v1`**, model **`inkling-2-mini-abliterated`**
(`inkling-2-mini` when serving without the ablation). No API key is required, and **there is no
authentication**: keep the server on loopback. Requests run one at a time; up to 4 more wait in a queue and
further ones get HTTP 503. `GET /health` and `GET /v1/models` report the state.

A follow-up request only reads what is new in its conversation. Conversations that take turns (subagents, a
second client) are kept warm in pinned host RAM, sized automatically (see `--park-host-mb` above)
(`--park-host-mb N` sets it in MiB, `0` turns it off); the least recently used is dropped when it is full.

`POST /v1/chat/completions` takes:

* `reasoning_effort`: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` (the chat template maps them to
  0, 0.1, 0.2, 0.7, 0.9, 0.99, 0.99) or a number from 0 to 0.99; the template's default is 0.9.
* `enable_thinking: false` to answer without reasoning (also accepted in `chat_template_kwargs`).
* `tools` (function calling through the model's chat template), `stop`, `max_tokens` / `max_completion_tokens`,
  `temperature`, `top_p`, `top_k`, `seed`. Requests that omit the sampling fields use the model's defaults
  (temperature 1, no cutoff; 0 is greedy) or the server's `--temperature` / `--top-p` / `--top-k`.
* `stream: true` for server-sent events; reasoning arrives as `reasoning_content` deltas before the answer's
  `content` (non-streaming responses carry `message.reasoning_content`). `stream_options.include_usage` adds usage.
* Images in user and tool messages, as content parts with a base64 `data:` URI (PNG, JPEG, WebP, GIF or BMP):
  `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`; `input_image` parts and Anthropic's
  `{"type": "image", "source": {"type": "base64", "data": ...}}` are accepted too. URLs are not fetched. See
  [Images](#images).

```bash
curl -s http://127.0.0.1:8003/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "inkling-2-mini-abliterated",
  "messages": [{"role": "user", "content": "Explain how hash maps handle collisions."}],
  "reasoning_effort": "low",
  "max_tokens": 4096
}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8003/v1", api_key="none")
stream = client.chat.completions.create(
    model="inkling-2-mini-abliterated",
    messages=[{"role": "user", "content": "Write a Python LRU cache with tests."}],
    reasoning_effort="medium",
    stream=True,
    # extra_body={"enable_thinking": False},  # answer without reasoning
)
for chunk in stream:
    delta = chunk.choices[0].delta
    print(getattr(delta, "reasoning_content", None) or delta.content or "", end="", flush=True)
```

### Images

The model reads images through its own vision tower, which is part of the checkpoint (no extra download). An
image costs **one token per 40x40 pixels**, `ceil(height / 40) * ceil(width / 40)`: 432 tokens for 960x720,
1,296 for 1920x1080. The model takes pixels as they are, without resizing; an image above 4,096 tokens (about
2560x2560, or a 4K screenshot) is reduced to fit, keeping its aspect ratio (`--image-max-tokens N`, up to the
model's 10,609 for 4096x4096 pixels). Transparency is shown on a grey checkerboard, as the model was trained.

```bash
curl -s http://127.0.0.1:8003/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "inkling-2-mini-abliterated",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,'"$(base64 -w0 picture.png)"'"}},
    {"type": "text", "text": "What is in this picture?"}]}],
  "max_tokens": 1024
}'
```

A request may hold 16 images and 32,768 image tokens (`--max-images`, `--max-image-tokens`); `usage.prompt_tokens`
counts image tokens. An image is read once per conversation: follow-up requests reuse it like any other part of
the prompt, and the same picture sent again is recognised by its pixels. `--no-vision` (in `INGINE_EXTRA_ARGS`)
serves text only and frees 0.6 GiB of GPU memory.

## Oh My Pi (OMP)

Merge this provider into `~/.omp/agent/models.yml`, keeping existing providers:

```yaml
providers:
  ingine:
    baseUrl: http://127.0.0.1:8003/v1
    auth: none
    api: openai-completions
    models:
      - id: inkling-2-mini-abliterated
        name: Inkling 2 Mini abliterated 256K (ingine)
        contextWindow: 262144
        maxTokens: 32768
        reasoning: true
        thinking:
          mode: effort
          efforts: [minimal, low, medium, high, xhigh, max]
          defaultLevel: medium
          requiresEffort: false
        input: [text, image]
        cost: {input: 0, output: 0, cacheRead: 0, cacheWrite: 0}
        compat:
          supportsDeveloperRole: true
          supportsStore: false
          supportsStrictMode: false
          supportsToolChoice: false
          supportsReasoningEffort: true
          reasoningEffortMap: {minimal: none}
          reasoningContentField: reasoning_content
          maxTokensField: max_tokens
          streamIdleTimeoutMs: 600000
```

Start the server first (`ingine start`), then open OMP:

```bash
omp --model ingine/inkling-2-mini-abliterated
```

OMP's `minimal` level is sent as `none` (thinking effort 0). If you lower `INGINE_CTX`, lower `contextWindow` to
match.

## Results

### Abliteration

Refusals on harmful test prompts (greedy decoding; an answer counts as a refusal when it contains a refusal phrase,
and every answer is in [`results/verify-abliterated.json`](results/verify-abliterated.json)):

| | original | abliterated |
|---|---|---|
| `harmful_test` (104), direct answer | 104/104 | 1/104 |
| JailbreakBench harmful behaviours (100), direct answer | 100/100 | 5/100 |
| `harmful_test` (48), reasoning first (effort low, 768-token budget) | 48/48 | 3/48 (each says what it cannot literally do, such as publish or spread the text, then writes it) |
| through ingine: `harmful_test` (104), direct answer | 104/104 | 3/104 |
| through ingine: the same with reasoning effort `none` | 104/104 | 0/104 |
| through ingine: `harmful_test` (48), reasoning first (effort low) | 48/48 | 1/48 (answers after the caveat) |
| through ingine: `harmful_test` (24), reasoning first (default effort) | 24/24 | 0/24 |
| through ingine: JailbreakBench (100), direct answer | 100/100 | 5/100 |

The first three rows are measured in the reference model in `tools/`. Not every prompt was unseen: the weight per
layer comes from an earlier search that scored refusals on prompts among the first 64 of `harmful_test`, and 10
JailbreakBench prompts occur in data some step used. On the prompts that nothing used, the abliterated model
refuses 0 of the other 40 `harmful_test` prompts and 4 of 90 JailbreakBench prompts, in the reference model and
through ingine alike.

Damage against the original model: KL divergence 0.014 on its own answers to 64 harmless prompts (the same earlier
search minimised KL on these prompts); on 8,192 tokens of technical prose that no step used, mean KL 0.005 and 96.4%
top-1 agreement.

Through ingine (32 borderline prompts and a small capability set, greedy):

| | original | abliterated |
|---|---|---|
| refusals, answering directly | 15/32 | 1/32 (the reply hedges, then writes a toned-down scene) |
| refusals, reasoning first | 8/32 | 0/32 |
| math (20, reasoning, effort low) | 20/20 | 20/20 |
| code (12, tests executed) | 12/12 | 12/12 |
| multiple choice (12) | 12/12 | 12/12 |

Reasoning stability with the default profile, sampled at temperature 1 (GSM8K: 40 problems at two reasoning
efforts; AIME 2024: 15 problems, effort high, up to 32,768 tokens):

| | original | abliterated |
|---|---|---|
| GSM8K, effort low: correct | 40/40 | 40/40 |
| GSM8K, effort high: correct | 40/40 | 40/40 |
| AIME 2024: correct | 13/15 | 14/15 |
| AIME 2024: ran out of tokens | 2/15 | 1/15 |
| reasoning that loops (95 samples) | 0 | 0 |

Repeating a request with the same seed returned the same text in 6 of 6 cases for each model. Raw data:
[`results/`](results).

### Speed

One RTX 5090, one request, default profile, tokens per second while generating. "chat" explains hash-map
collisions and "code" writes an LRU-cache module with tests (short prompts, up to 1024 tokens, median of 6 runs);
"coding agent" replays 24 recorded agent requests (2K-39K-token prompts with tools; mean of two replays):

| | chat | code | coding agent |
|---|---:|---:|---:|
| temperature 1 (the default) | 487 | 554 | 501 |
| greedy (temperature 0) | 501 | 561 | 510 |

The default profile runs DSpark speculative decoding: a small drafter proposes tokens and the full model checks
every one, so it changes the speed, not the output distribution. Predictable text goes fastest: in the agent replay,
tool calls run at about 740 tokens/s, reasoning and prose at about 450; text outside English and code gains the least.

A new prompt is read at 10,900 tokens/s (20K tokens), 9,600 (64K), 8,100 (131K) and 7,000 (200K); a short prompt
takes 0.09 s to the first token. The server holds 24.4 GiB of GPU memory after start-up (23.8 GiB with
`--no-vision`) and no more while reading a 254K-token prompt; conversations kept warm add up to 3 GiB.

Images: the vision tower takes 3 ms for an image of up to 64 tokens (320x320 pixels), 21 ms for 576 tokens
(960x960), 0.17 s for 1,849 tokens (1720x1720) and 1.1 s for 4,096 tokens, using up to 0.85 GiB of working memory
for the largest; its tokens are then read like text. A 3840x2160 screenshot (reduced to 4,080 tokens) is answered
1.5 s after the request. Against the transformers reference in FP32 the image embeddings have a cosine of 0.9995
to 0.99999, closer than transformers' own BF16 model (`native/tests/vision`).

### Agent loops

Twelve recorded Oh My Pi sessions (230 requests, prompts of 2K-89K tokens, 6.9M prompt tokens in total), replayed
in order (greedy, up to 400 tokens per answer): 214 of the 230 requests continue a conversation the server still
holds and read only what is new, so all 230 together wait 50 s for their first tokens; decoding runs at 563
tokens/s. Two of these sessions taking turns, as an agent and its subagent do (67 requests), wait 8.5 s in total.

### Long context

The default profile serves **262,144 tokens** of context (the checkpoint supports 1M). The long-range attention
layers keep their cache in FP8, 16 KiB per token (4 GiB at 262,144 tokens, allocated at start-up), so that it fits
beside the model in 32 GB; the sliding-window layers keep their last 1,024 tokens in BF16. A code haystack with a
passphrase hidden in the middle:

| | 131,443-token prompt | 254,019-token prompt |
|---|---:|---:|
| reading the prompt | 16.4 s (8,000 tokens/s) | 40.9 s (6,200 tokens/s) |
| decoding afterwards | 354 tokens/s | 258 tokens/s |

The passphrase was found in both. With an image after 199,000 tokens of text, the model answers from both: the
passphrase at the start of the text and the words written in the image (26.5 s).

## License and credits

This repository (ingine, tools, installer and the abliteration file) is licensed under **Apache-2.0**, see
[LICENSE](LICENSE). The base model, [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) by
Thinking Machines, is also Apache-2.0; its files are redistributed unchanged in the Hugging Face repository
[satellitedown/Inkling-2-Mini-abliterated-ingine](https://huggingface.co/satellitedown/Inkling-2-Mini-abliterated-ingine),
not in this Git repository. CUDA components keep NVIDIA's license terms.

The search follows [Heretic](https://github.com/p-e-w/heretic) by p-e-w and the abliteration literature; the
reference model is built on the transformers Inkling2Loop pull request. See [NOTICE](NOTICE). The abliteration
removes the model's safety refusals: you are responsible for how you use it. This is an unofficial community
project, not endorsed by Thinking Machines.
