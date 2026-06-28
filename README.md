# qwen-vl

A dependency-free Python CLI that turns an image into text using an
[Ollama](https://ollama.com)-hosted Qwen vision model. Uses only the Python
standard library — it talks to the local Ollama REST API directly.

## Prerequisites

- Ollama running locally (`ollama serve`)
- A Qwen vision model pulled, e.g.:

  ```sh
  ollama pull qwen2.5vl:7b
  ```

## Usage

```sh
# Describe / transcribe an image (streams to stdout)
python3 qwen_vl.py photo.jpg

# Custom prompt (e.g. pure OCR)
python3 qwen_vl.py receipt.png -p "Transcribe all text exactly, preserving line breaks."

# Pick a different model
python3 qwen_vl.py photo.jpg -m qwen2.5vl:7b

# Save to a file instead of stdout
python3 qwen_vl.py scan.png -o out.txt

# Talk to a remote Ollama host
python3 qwen_vl.py photo.jpg --host http://192.168.1.10:11434
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `image` | Path to the image to process | (required) |
| `-m, --model` | Ollama model name | `qwen2.5vl:7b` |
| `-p, --prompt` | Instruction sent with the image | describe + transcribe |
| `--host` | Ollama host URL | `http://localhost:11434` |
| `-o, --output` | Write result to a file (disables streaming) | stdout |
| `--no-stream` | Wait for full response instead of streaming | off |
| `--timeout` | Request timeout in seconds | `300` |
