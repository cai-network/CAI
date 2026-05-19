<!--
SPDX-FileCopyrightText: 2025 cai Technologies Ltd
SPDX-FileCopyrightText: 2026 CAI contributors
SPDX-License-Identifier: Apache-2.0
-->
<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/imgs/CAI-logo-black-bg.jpg">
  <img alt="cai logo" src="/docs/imgs/CAI-logo-transparent.png" width="50%" height="50%">
</picture>

cai: Run frontier AI locally. Maintained by [cai labs](https://x.com/CAIlabs).

<p align="center">
  <a href="https://discord.gg/TJ4P57arEm" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Discord-Join%20Server-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://x.com/CAIlabs" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/twitter/follow/CAIlabs?style=social" alt="X"></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0.html" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/License-Apache2.0-blue.svg" alt="License: Apache-2.0"></a>
</p>

</div>

---

cai connects all your devices into an AI cluster. Not only does cai enable running models larger than would fit on a single device, but with [day-0 support for RDMA over Thunderbolt](https://x.com/CAIlabs/status/2001817749744476256?s=20), makes models run faster as you add more devices.

## Features

- **Automatic Device Discovery**: Devices running cai automatically discover each other - no manual configuration.
- **RDMA over Thunderbolt**: cai ships with [day-0 support for RDMA over Thunderbolt 5](https://x.com/CAIlabs/status/2001817749744476256?s=20), enabling 99% reduction in latency between devices.
- **Topology-Aware Auto Parallel**: cai figures out the best way to split your model across all available devices based on a realtime view of your device topology. It takes into account device resources and network latency/bandwidth between each link.
- **Tensor Parallelism**: cai supports sharding models, for up to 1.8x speedup on 2 devices and 3.2x speedup on 4 devices.
- **MLX Support**: cai uses [MLX](https://github.com/ml-explore/mlx) as an inference backend and [MLX distributed](https://ml-explore.github.io/mlx/build/html/usage/distributed.html) for distributed communication.
- **Multiple API Compatibility**: Compatible with OpenAI Chat Completions API, Claude Messages API, OpenAI Responses API, and Ollama API - use your existing tools and clients.
- **Custom Model Support**: Load custom models from HuggingFace hub to expand the range of available models.

## Dashboard

cai includes a built-in dashboard for managing your cluster and chatting with models.

<p align="center">
  <img src="docs/imgs/dashboard-cluster-view.png" alt="cai dashboard - cluster view showing 4 x M3 Ultra Mac Studio with DeepSeek v3.1 and Kimi-K2-Thinking loaded" width="80%" />
</p>
<p align="center"><em>4 × 512GB M3 Ultra Mac Studio running DeepSeek v3.1 (8-bit) and Kimi-K2-Thinking (4-bit)</em></p>

## Benchmarks

<details>
  <summary>Qwen3-235B (8-bit) on 4 × M3 Ultra Mac Studio with Tensor Parallel RDMA</summary>
  <img src="docs/benchmarks/jeffgeerling/mac-studio-cluster-ai-full-1-qwen3-235b.jpeg" alt="Benchmark - Qwen3-235B (8-bit) on 4 × M3 Ultra Mac Studio with Tensor Parallel RDMA" width="80%" />
  <p>
    <strong>Source:</strong> <a href="https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5">Jeff Geerling: 15 TB VRAM on Mac Studio – RDMA over Thunderbolt 5</a>
  </p>
</details>

<details>
  <summary>DeepSeek v3.1 671B (8-bit) on 4 × M3 Ultra Mac Studio with Tensor Parallel RDMA</summary>
  <img src="docs/benchmarks/jeffgeerling/mac-studio-cluster-ai-full-2-deepseek-3.1-671b.jpeg" alt="Benchmark - DeepSeek v3.1 671B (8-bit) on 4 × M3 Ultra Mac Studio with Tensor Parallel RDMA" width="80%" />
  <p>
    <strong>Source:</strong> <a href="https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5">Jeff Geerling: 15 TB VRAM on Mac Studio – RDMA over Thunderbolt 5</a>
  </p>
</details>

<details>
  <summary>Kimi K2 Thinking (native 4-bit) on 4 × M3 Ultra Mac Studio with Tensor Parallel RDMA</summary>
  <img src="docs/benchmarks/jeffgeerling/mac-studio-cluster-ai-full-3-kimi-k2-thinking.jpeg" alt="Benchmark - Kimi K2 Thinking (native 4-bit) on 4 × M3 Ultra Mac Studio with Tensor Parallel RDMA" width="80%" />
  <p>
    <strong>Source:</strong> <a href="https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5">Jeff Geerling: 15 TB VRAM on Mac Studio – RDMA over Thunderbolt 5</a>
  </p>
</details>

---

## Quick Start

Devices running cai automatically discover each other, without needing any manual configuration. Each device provides an API and a dashboard for interacting with your cluster (runs at `http://localhost:52415`).

There are two ways to run cai:

### Run from Source (macOS)

If you have [Nix](https://nixos.org/) installed, you can skip most of the steps below and run cai directly:

```bash
nix run .#cai
```

**Note:** To accept the Cachix binary cache (and avoid the Xcode Metal ToolChain), add to `/etc/nix/nix.conf`:
```
trusted-users = root    (or your username)
experimental-features = nix-command flakes
```
Then restart the Nix daemon: `sudo launchctl kickstart -k system/org.nixos.nix-daemon`

**Prerequisites:**
- [Xcode](https://developer.apple.com/xcode/) (provides the Metal ToolChain required for MLX compilation)
- [brew](https://github.com/Homebrew/brew) (for simple package management on macOS)

  ```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  ```
- [uv](https://github.com/astral-sh/uv) (for Python dependency management)
- [node](https://github.com/nodejs/node) (for building the dashboard)

  ```bash
  brew install uv node
  ```
- [rust](https://github.com/rust-lang/rustup) (to build Rust bindings, nightly for now)

  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  rustup toolchain install nightly
  ```
- [macmon](https://github.com/vladkens/macmon) (for hardware monitoring on Apple Silicon)

  Install the pinned fork revision used by this repo instead of Homebrew `macmon`.
  Homebrew `macmon 0.6.1` still crashes on Apple M5.

  ```bash
  cargo install --git https://github.com/vladkens/macmon \
    --rev a1cd06b6cc0d5e61db24fd8832e74cd992097a7d \
    macmon \
    --force
  ```

Clone the repo, build the dashboard, and run cai:

```bash
# Clone cai
git clone https://github.com/CAI-explore/cai

# Build dashboard
cd CAI/dashboard && npm install && npm run build && cd ..

# Run cai
uv run cai
```

This starts the cai dashboard and API at http://localhost:52415/


*Please view the section on RDMA to enable this feature on MacOS >=26.2!*


### Run from Source (Linux)

**Prerequisites:**

- [uv](https://github.com/astral-sh/uv) (for Python dependency management)
- [node](https://github.com/nodejs/node) (for building the dashboard) - version 18 or higher
- [rust](https://github.com/rust-lang/rustup) (to build Rust bindings, nightly for now)

**Installation methods:**

**Option 1: Using system package manager (Ubuntu/Debian example):**
```bash
# Install Node.js and npm
sudo apt update
sudo apt install nodejs npm

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Rust (using rustup)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup toolchain install nightly
```

**Option 2: Using Homebrew on Linux (if preferred):**
```bash
# Install Homebrew on Linux
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install uv node

# Install Rust (using rustup)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup toolchain install nightly
```

**Note:** The `macmon` package is macOS-only and not required for Linux.

Clone the repo, build the dashboard, and run cai:

```bash
# Clone cai
git clone https://github.com/CAI-explore/cai

# Build dashboard
cd CAI/dashboard && npm install && npm run build && cd ..

# Run cai
uv run cai
```

This starts the cai dashboard and API at http://localhost:52415/

**Important note for Linux users:** Currently, cai runs on CPU on Linux. GPU support for Linux platforms is under development. If you'd like to see support for your specific Linux hardware, please [search for existing feature requests](https://github.com/CAI-explore/CAI/issues) or create a new one.

**Configuration Options:**

- `--no-worker`: Run cai without the worker component. Useful for coordinator-only nodes that handle networking and orchestration but don't execute inference tasks. This is helpful for machines without sufficient GPU resources but with good network connectivity.

  ```bash
  uv run cai --no-worker
  ```

**File Locations (Linux):**

cai follows the [XDG Base Directory Specification](https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html) on Linux:

- **Configuration files**: `~/.config/CAI/` (or `$XDG_CONFIG_HOME/CAI/`)
- **Data files**: `~/.local/share/CAI/` (or `$XDG_DATA_HOME/CAI/`)
- **Cache files**: `~/.cache/CAI/` (or `$XDG_CACHE_HOME/CAI/`)
- **Log files**: `~/.cache/CAI/CAI_log/` (with automatic log rotation)
- **Custom model cards**: `~/.local/share/CAI/custom_model_cards/`

You can override these locations by setting the corresponding XDG environment variables.

### macOS App

cai ships a macOS app that runs in the background on your Mac.

<img src="docs/imgs/macos-app-one-macbook.png" alt="cai macOS App - running on a MacBook" width="35%" />

The macOS app requires macOS Tahoe 26.2 or later.

Download the latest build here: [CAI-latest.dmg](https://assets.CAIlabs.net/CAI-latest.dmg).

The app will ask for permission to modify system settings and install a new Network profile. Improvements to this are being worked on.

**Custom Namespace for Cluster Isolation:**

The macOS app includes a custom namespace feature that allows you to isolate your cai cluster from others on the same network. This is configured through the `CAI_LIBP2P_NAMESPACE` setting:

- **Use cases**:
  - Running multiple separate cai clusters on the same network
  - Isolating development/testing clusters from production clusters
  - Preventing accidental cluster joining

- **Configuration**: Access this setting in the app's Advanced settings (or set the `CAI_LIBP2P_NAMESPACE` environment variable when running from source)

The namespace is logged on startup for debugging purposes.

#### Uninstalling the macOS App

The recommended way to uninstall is through the app itself: click the menu bar icon → Advanced → Uninstall. This cleanly removes all system components.

If you've already deleted the app, you can run the standalone uninstaller script:

```bash
sudo ./app/CAI/uninstall-cai.sh
```

This removes:
- Network setup LaunchDaemon
- Network configuration script
- Log files
- The "cai" network location

**Note:** You'll need to manually remove cai from Login Items in System Settings → General → Login Items.

---

### Enabling RDMA on macOS

RDMA is a new capability added to macOS 26.2. It works on any Mac with Thunderbolt 5 (M4 Pro Mac Mini, M4 Max Mac Studio, M4 Max MacBook Pro, M3 Ultra Mac Studio).

Please refer to the caveats for immediate troubleshooting.

To enable RDMA on macOS, follow these steps:

1. Shut down your Mac.
2. Hold down the power button for 10 seconds until the boot menu appears.
3. Select "Options" to enter Recovery mode.
4. When the Recovery UI appears, open the Terminal from the Utilities menu.
5. In the Terminal, type:
   ```
   rdma_ctl enable
   ```
   and press Enter.
6. Reboot your Mac.

After that, RDMA will be enabled in macOS and cai will take care of the rest.

**Important Caveats**

1. Devices that wish to be part of an RDMA cluster must be connected to all other devices in the cluster.
2. The cables must support TB5.
3. On a Mac Studio, you cannot use the Thunderbolt 5 port next to the Ethernet port.
4. If running from source, please use the script found at `tmp/set_rdma_network_config.sh`, which will disable Thunderbolt Bridge and set dhcp on each RDMA port.
5. RDMA ports may be unable to discover each other on different versions of MacOS. Please ensure that OS versions match exactly (even beta version numbers) on all devices.

---

## Environment Variables

cai supports several environment variables for configuration:

| Variable | Description | Default |
|----------|-------------|---------|
| `CAI_DEFAULT_MODELS_DIR` | Default directory for model downloads and caches. Always first in the writable dirs list. | `~/.local/share/CAI/models` (Linux) or `~/.CAI/models` (macOS) |
| `CAI_MODELS_DIRS` | Colon-separated additional writable directories for model downloads. Checked in order after the default; first with enough free space is used. | None |
| `CAI_MODELS_READ_ONLY_DIRS` | Colon-separated read-only directories to search for pre-downloaded models (e.g., NFS mounts, shared storage). Models here cannot be deleted. | None |
| `CAI_OFFLINE` | Run without internet connection (uses only local models) | `false` |
| `CAI_ENABLE_IMAGE_MODELS` | Enable image model support | `false` |
| `CAI_LIBP2P_NAMESPACE` | Custom namespace for cluster isolation | None |
| `CAI_FAST_SYNCH` | Control MLX_METAL_FAST_SYNCH behavior (for JACCL backend) | Auto |
| `CAI_TRACING_ENABLED` | Enable distributed tracing for performance analysis | `false` |

**Example usage:**

```bash
# Use pre-downloaded models from NFS mount (read-only)
CAI_MODELS_READ_ONLY_DIRS=/mnt/nfs/models:/opt/ai-models uv run cai

# Download models to an external SSD (falls back to default dir if full)
CAI_MODELS_DIRS=/Volumes/ExternalSSD/CAI-models uv run cai

# Run in offline mode
CAI_OFFLINE=true uv run cai

# Enable image models
CAI_ENABLE_IMAGE_MODELS=true uv run cai

# Use custom namespace for cluster isolation
CAI_LIBP2P_NAMESPACE=my-dev-cluster uv run cai
```

---

### Using the API

cai provides multiple API-compatible interfaces for maximum compatibility with existing tools:

- **OpenAI Chat Completions API** - Compatible with OpenAI clients
- **Claude Messages API** - Compatible with Anthropic's Claude format
- **OpenAI Responses API** - Compatible with OpenAI's Responses format
- **Ollama API** - Compatible with Ollama and tools like OpenWebUI

If you prefer to interact with cai via the API, here is an example creating an instance of a small model (`mlx-community/Llama-3.2-1B-Instruct-4bit`), sending a chat completions request and deleting the instance.

---

**1. Preview instance placements**

The `/instance/previews` endpoint will preview all valid placements for your model.

```bash
curl "http://localhost:52415/instance/previews?model_id=llama-3.2-1b"
```

Sample response:

```json
{
  "previews": [
    {
      "model_id": "mlx-community/Llama-3.2-1B-Instruct-4bit",
      "sharding": "Pipeline",
      "instance_meta": "MlxRing",
      "instance": {...},
      "memory_delta_by_node": {"local": 729808896},
      "error": null
    }
    // ...possibly more placements...
  ]
}
```

This will return all valid placements for this model. Pick a placement that you like.
To pick the first one, pipe into `jq`:

```bash
curl "http://localhost:52415/instance/previews?model_id=llama-3.2-1b" | jq -c '.previews[] | select(.error == null) | .instance' | head -n1
```

---

**2. Create a model instance**

Send a POST to `/instance` with your desired placement in the `instance` field (the full payload must match types as in `CreateInstanceParams`), which you can copy from step 1:

```bash
curl -X POST http://localhost:52415/instance \
  -H 'Content-Type: application/json' \
  -d '{
    "instance": {...}
  }'
```


Sample response:

```json
{
  "message": "Command received.",
  "command_id": "e9d1a8ab-...."
}
```

---

**3. Send a chat completion**

Now, make a POST to `/v1/chat/completions` (the same format as OpenAI's API):

```bash
curl -N -X POST http://localhost:52415/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [
      {"role": "user", "content": "What is Llama 3.2 1B?"}
    ],
    "stream": true
  }'
```

---

**4. Delete the instance**

When you're done, delete the instance by its ID (find it via `/state` or `/instance` endpoints):

```bash
curl -X DELETE http://localhost:52415/instance/YOUR_INSTANCE_ID
```

### Claude Messages API Compatibility

Use the Claude Messages API format with the `/v1/messages` endpoint:

```bash
curl -N -X POST http://localhost:52415/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [
      {"role": "user", "content": "Hello"}
    ],
    "max_tokens": 1024,
    "stream": true
  }'
```

### OpenAI Responses API Compatibility

Use the OpenAI Responses API format with the `/v1/responses` endpoint:

```bash
curl -N -X POST http://localhost:52415/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [
      {"role": "user", "content": "Hello"}
    ],
    "stream": true
  }'
```

### Ollama API Compatibility

cai supports Ollama API endpoints for compatibility with tools like OpenWebUI:

```bash
# Ollama chat
curl -X POST http://localhost:52415/ollama/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [
      {"role": "user", "content": "Hello"}
    ],
    "stream": false
  }'

# List models (Ollama format)
curl http://localhost:52415/ollama/api/tags
```

### Custom Model Loading from HuggingFace

You can add custom models from the HuggingFace hub:

```bash
curl -X POST http://localhost:52415/models/add \
  -H 'Content-Type: application/json' \
  -d '{
    "model_id": "mlx-community/my-custom-model"
  }'
```

**Security Note:**

Custom models requiring `trust_remote_code` in their configuration must be explicitly enabled (default is false) for security. Only enable this if you trust the model's remote code execution. Models are fetched from HuggingFace and stored locally as custom model cards.

**Other useful API endpoints*:**

- List all models: `curl http://localhost:52415/models`
- List downloaded models only: `curl http://localhost:52415/models?status=downloaded`
- Search HuggingFace: `curl "http://localhost:52415/models/search?query=llama&limit=10"`
- Inspect instance IDs and deployment state: `curl http://localhost:52415/state`

For further details, see:

- API documentation in [docs/api.md](docs/api.md).
- API types and endpoints in [src/CAI/master/api.py](src/CAI/master/api.py).

---

## Benchmarking

The `cai-bench` tool measures model prefill and token generation speed across different placement configurations. This helps you optimize model performance and validate improvements.

**Prerequisites:**
- Nodes should be running with `uv run cai` before benchmarking
- The tool uses the `/bench/chat/completions` endpoint

**Basic usage:**

```bash
uv run bench/cai_bench.py \
  --model Llama-3.2-1B-Instruct-4bit \
  --pp 128,256,512 \
  --tg 128,256
```

**Key parameters:**

- `--model`: Model to benchmark (short ID or HuggingFace ID)
- `--pp`: Prompt size hints (comma-separated integers)
- `--tg`: Generation lengths (comma-separated integers)
- `--max-nodes`: Limit placements to N nodes (default: 4)
- `--instance-meta`: Filter by `ring`, `jaccl`, or `both` (default: both)
- `--sharding`: Filter by `pipeline`, `tensor`, or `both` (default: both)
- `--repeat`: Number of repetitions per configuration (default: 1)
- `--warmup`: Warmup runs per placement (default: 0)
- `--json-out`: Output file for results (default: bench/results.json)

**Example with filters:**

```bash
uv run bench/cai_bench.py \
  --model Llama-3.2-1B-Instruct-4bit \
  --pp 128,512 \
  --tg 128 \
  --max-nodes 2 \
  --sharding tensor \
  --repeat 3 \
  --json-out my-results.json
```

The tool outputs performance metrics including prompt tokens per second (prompt_tps), generation tokens per second (generation_tps), and peak memory usage for each configuration.

---

## Hardware Accelerator Support

On macOS, cai uses the GPU. On Linux, cai currently runs on CPU. We are working on extending hardware accelerator support. If you'd like support for a new hardware platform, please [search for an existing feature request](https://github.com/CAI-explore/CAI/issues) and add a thumbs up so we know what hardware is important to the community.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to cai.

