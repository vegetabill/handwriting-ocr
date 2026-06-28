#!/usr/bin/env python3
"""Process an image into text using an Ollama-hosted Qwen vision model.

Dependency-free: talks to the local Ollama REST API over stdlib only.
"""
import argparse
import base64
import json
import sys
import urllib.error
import urllib.request

DEFAULT_MODEL = "qwen2.5vl:7b"
DEFAULT_PROMPT = "Describe this image in detail. Transcribe any text you see verbatim."


def encode_image(path):
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def generate(host, model, prompt, image_b64, stream, timeout):
    url = host.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": stream,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )

    parts = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if stream:
            for line in resp:
                line = line.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    raise RuntimeError(chunk["error"])
                token = chunk.get("response", "")
                parts.append(token)
                sys.stdout.write(token)
                sys.stdout.flush()
            sys.stdout.write("\n")
        else:
            body = json.loads(resp.read())
            if "error" in body:
                raise RuntimeError(body["error"])
            parts.append(body.get("response", ""))
    return "".join(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert an image to text via an Ollama Qwen vision model."
    )
    parser.add_argument("image", help="Path to the image file to process")
    parser.add_argument(
        "-m", "--model", default=DEFAULT_MODEL,
        help=f"Ollama model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "-p", "--prompt", default=DEFAULT_PROMPT,
        help="Instruction prompt sent with the image",
    )
    parser.add_argument(
        "--host", default="http://127.0.0.1:11434",
        help="Ollama host URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "-o", "--output", help="Write result to this file instead of stdout"
    )
    parser.add_argument(
        "--no-stream", action="store_true",
        help="Disable token streaming; wait for the full response",
    )
    parser.add_argument(
        "--timeout", type=float, default=300.0,
        help="Request timeout in seconds (default: 300)",
    )
    args = parser.parse_args(argv)

    try:
        image_b64 = encode_image(args.image)
    except OSError as exc:
        parser.error(f"cannot read image: {exc}")

    stream = not args.no_stream and not args.output

    try:
        text = generate(
            args.host, args.model, args.prompt, image_b64, stream, args.timeout
        )
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            body = json.loads(exc.read())
            detail = body.get("error", "")
            # Ollama sometimes double-encodes the error as JSON inside "error".
            if detail.startswith("{"):
                detail = json.loads(detail).get("error", {}).get("message", detail)
        except (ValueError, AttributeError):
            pass
        msg = f"error: Ollama returned HTTP {exc.code}"
        if detail:
            msg += f": {detail}"
        if exc.code == 400 and "multimodal" in detail.lower():
            msg += (
                f"\nhint: '{args.model}' is a text-only model. Use a vision "
                "model, e.g. -m qwen2.5vl:7b"
            )
        elif exc.code == 404:
            msg += f"\nhint: pull it first with `ollama pull {args.model}`"
        sys.exit(msg)
    except urllib.error.URLError as exc:
        sys.exit(f"error: cannot reach Ollama at {args.host}: {exc}")
    except RuntimeError as exc:
        sys.exit(f"error: {exc}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {len(text)} chars to {args.output}", file=sys.stderr)
    elif args.no_stream:
        print(text)


if __name__ == "__main__":
    main()
