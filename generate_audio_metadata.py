"""
Audio Metadata Generator with Qwen2.5-Omni GPTQ Model (Debug‑enabled)

Additional, Non‑Optional Requirements:
- flash_attn
- qwen_omni_utils

Usable Environment Example:
- Arch Linux
- RTX 4070 Ti SUPER (16 GB VRAM)
- Python 3.10.17
- Reference `example_environment.txt` for installed packages

Usage (normal):
    CUDA_VISIBLE_DEVICES=0 python3 ./generate_audio_metadata_debug.py '{your_audio_directory}' --model-path Qwen/Qwen2.5-Omni-7B-GPTQ-Int4

Usage (with VRAM tracking):
    CUDA_VISIBLE_DEVICES=0 python3 ./generate_audio_metadata_debug.py '{your_audio_directory}' --debug-vram

Expected VRAM Usage: 
    average of 8 GB, spikes near the start between 12‑16 GB (optimization needed still)
"""
import logging
logging.getLogger("accelerate").setLevel(logging.ERROR)
from transformers.utils import logging
logging.set_verbosity_error()
import os
import sys
import gc
import datetime
import argparse
from typing import Dict, Any, Optional
import torch



# ────────────────────────── DEBUG HELPERS ────────────────────────── #
DEBUG_VRAM: bool = False  # set in main()


def _fmt_mb(bytes_val: int) -> float:
    return bytes_val / (1024 ** 2)


def log_vram(event: str) -> None:
    """Print a timestamped one‑liner with current & peak VRAM stats."""
    if not DEBUG_VRAM or not torch.cuda.is_available():
        return

    torch.cuda.synchronize()
    alloc = _fmt_mb(torch.cuda.memory_allocated())
    reserved = _fmt_mb(torch.cuda.memory_reserved())
    max_alloc = _fmt_mb(torch.cuda.max_memory_allocated())

    now = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] VRAM | {event:<40} | allocated={alloc:7.0f} MB | reserved={reserved:7.0f} MB | peak={max_alloc:7.0f} MB")


def reset_vram_peak() -> None:
    if DEBUG_VRAM and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ────────────────────────── GPTQ imports ────────────────────────── #
# shout‑out to the Qwen Team for making the sub‑directories contain hyphens! :)

here = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(here, "low-VRAM-mode", "modeling_qwen2_5_omni_low_VRAM_mode.py")

import importlib.util
spec = importlib.util.spec_from_file_location("modeling_qwen2_5_omni_low_VRAM_mode", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)  # type: ignore

Qwen2_5OmniForConditionalGeneration = module.Qwen2_5OmniForConditionalGeneration
from transformers import Qwen2_5OmniProcessor
from transformers.utils.hub import cached_file

from gptqmodel import GPTQModel
from gptqmodel.models.base import BaseGPTQModel
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models._const import CPU, SUPPORTED_MODELS
from huggingface_hub import snapshot_download

from qwen_omni_utils import process_mm_info

# Define the GPTQ model class
def move_to(module, device):
    """Helper function to move a module to a device"""
    if module is not None:
        return module.to(device)
    return module

class Qwen25OmniThinkerGPTQ(BaseGPTQModel):
    loader = Qwen2_5OmniForConditionalGeneration
    base_modules = [
        "thinker.model.embed_tokens", 
        "thinker.model.norm", 
        "thinker.audio_tower", 
        "thinker.model.rotary_emb"
    ]
    pre_lm_head_norm_module = "thinker.model.norm"
    require_monkeypatch = False
    layers_node = "thinker.model.layers"
    layer_type = "Qwen2_5OmniDecoderLayer"
    layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]
   
    def pre_quantize_generate_hook_start(self):
        self.disable_talker()
        self.thinker.audio_tower = move_to(self.thinker.audio_tower, device=self.quantize_config.device)

    def pre_quantize_generate_hook_end(self):
        self.thinker.audio_tower = move_to(self.thinker.audio_tower, device=CPU)

    def preprocess_dataset(self, sample: Dict) -> Dict:
        return sample

# Register the GPTQ model
MODEL_MAP["qwen2_5_omni"] = Qwen25OmniThinkerGPTQ
SUPPORTED_MODELS.extend(["qwen2_5_omni"])

# Patch the from_config method for speaker dictionary loading
@classmethod
def patched_from_config(cls, config, *args, **kwargs):
    kwargs.pop("trust_remote_code", None)
    model_path = kwargs.get("model_path", None)
    model = cls._from_config(config, **kwargs)
    return model

Qwen2_5OmniForConditionalGeneration.from_config = patched_from_config

# System prompt for audio analysis
# Note: Using a custom system prompt will trigger a warning about audio output mode,
# but this is fine since we're only generating text, not audio
SYSTEM_PROMPT = """Analyze the input audio and generate 6 description variants. Each variant must be <200 characters. Follow these exact definitions:

1.  `simplified`: Use only one most representative tag from the valid set.
2.  `expanded`: Broaden valid tags to include related sub-genres/techniques.
3.  `descriptive`: Convert tags into a sensory-rich sentence based *only on the sound*. DO NOT transcribe or reference the lyrics.
4.  `synonyms`: Replace tags with equivalent terms (e.g., 'strings' → 'orchestral').
5.  `use_cases`: Suggest practical applications based on audio characteristics.
6.  `analysis`: Analyze the audio's genre, instruments, tempo, and mood **based strictly on the audible musical elements**. Technical breakdown in specified format.
    *   For the `instruments` list: **Only include instruments that are actually heard playing in the audio recording.** **Explicitly ignore any instruments merely mentioned or sung about in the lyrics.** Cover all audibly present instruments.
7. `lyrical_rap_check`: if the audio is lyrical rap
**Strictly ignore any information derived solely from the lyrics when performing the analysis, especially for identifying instruments.**

**Output Format:**
```json
{
  "simplified": <str>,
  "expanded": <str>,
  "descriptive": <str>,
  "synonyms": <str>,
  "use_cases": <str>,
  "analysis": {
    "genre": <str list>,
    "instruments": <str list>,
    "tempo": <str>,
    "mood": <str list>
  },
  "lyrical_rap_check": <bool>
}
```"""


# ────────────────────────── Model loading ────────────────────────── #

def load_model(model_path: str, device_map: Optional[Dict[str, str]] = None):
    """Download (if needed) and load the GPTQ model + processor."""
    log_vram("load_model: start")

    # Local cache if HuggingFace repo
    if not os.path.exists(model_path):
        model_path = snapshot_download(repo_id=model_path)  # type: ignore

    if device_map is None:
        device_map = {
            "thinker.model": "cuda", 
            "thinker.lm_head": "cuda", 
            "thinker.visual": "cpu",  
            "thinker.audio_tower": "cpu",  
            "talker": "cuda",  
            "token2wav": "cuda",  
        }
    
    # Load GPTQ model
    model = GPTQModel.load(
        model_path, 
        device_map=device_map, 
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        parallel_packing=False,
        cache_block_outputs=False,
        true_sequential=True,
        model_path=model_path
    )
    log_vram("load_model: after GPTQModel.load()")

    # Try disabling audio output to avoid extra buffers
    try:
        if hasattr(model, "disable_talker"):
            model.disable_talker()
    except Exception as e:
        print(f"Note: Could not disable audio components: {e}")

    # Keep visual/audio tower off‑GPU by default
    if hasattr(model, "thinker"):
        if getattr(model.thinker, "visual", None) is not None:
            model.thinker.visual = model.thinker.visual.to("cpu")
        if getattr(model.thinker, "audio_tower", None) is not None:
            model.thinker.audio_tower = model.thinker.audio_tower.to("cpu")

    torch.cuda.empty_cache()
    log_vram("load_model: ready (after empty_cache)")
    reset_vram_peak()

    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    return model, processor


# ────────────────────────── Audio analysis ────────────────────────── #

def analyze_audio(audio_path: str, model, processor, *, system_prompt: str = SYSTEM_PROMPT):
    """Run one inference; heavy blocks are wrapped with VRAM logging."""

    log_vram(f"analyze_audio: {os.path.basename(audio_path)} — start")
    reset_vram_peak()

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "audio", "audio": audio_path}]},
    ]

    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    log_vram("inputs prepared (CPU)")

    # Activate audio tower on GPU only when required
    if hasattr(model, "thinker") and getattr(model.thinker, "audio_tower", None) is not None:
        model.thinker.audio_tower = model.thinker.audio_tower.to("cuda")
    log_vram("audio_tower → GPU")

    # Copy tensors to GPU and match dtypes
    inputs_cuda: Dict[str, Any] = {}
    for k, v in inputs.items():
        if hasattr(v, "to"):
            inputs_cuda[k] = v.to("cuda")
            if hasattr(model, "dtype") and v.dtype.is_floating_point:
                inputs_cuda[k] = inputs_cuda[k].to(model.dtype)
        else:
            inputs_cuda[k] = v
    log_vram("inputs → GPU")

    # ── Generation ──────────────────────────────────────────────
    with torch.no_grad():
        output_ids = (
            model.thinker.generate(
                **inputs_cuda,
                max_new_tokens=4096,
                use_audio_in_video=False,
            )
            if hasattr(model, "thinker")
            else model.generate(  # type: ignore
                **inputs_cuda,
                return_audio=False,
                use_audio_in_video=False,
            )
        )
    log_vram("generation finished")

    # Move optional towers back to CPU asap
    if hasattr(model, "thinker"):
        if getattr(model.thinker, "visual", None) is not None:
            model.thinker.visual = model.thinker.visual.to("cpu")
        if getattr(model.thinker, "audio_tower", None) is not None:
            model.thinker.audio_tower = model.thinker.audio_tower.to("cpu")
    torch.cuda.empty_cache()
    log_vram("audio_tower → CPU & cache cleared")

    generate_ids = output_ids[:, inputs.input_ids.size(1) :]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    log_vram("decode done / end")
    reset_vram_peak()
    return response


# ────────────────────────── CLI entry‑point ────────────────────────── #

def main() -> None:
    global DEBUG_VRAM  # noqa: PLW0603

    parser = argparse.ArgumentParser(description="Analyze audio directory with Qwen2.5‑Omni GPTQ model (with optional VRAM debug)")
    parser.add_argument("audio_dir", type=str, help="Path to directory containing audio files")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-Omni-7B-GPTQ-Int4", help="Model path or HF repo")
    parser.add_argument("--system-prompt", type=str, default=None, help="Custom system prompt override")
    parser.add_argument("--debug-vram", action="store_true", help="Print VRAM usage at key steps")
    args = parser.parse_args()

    DEBUG_VRAM = args.debug_vram or bool(os.environ.get("DEBUG_VRAM"))

    if not os.path.isdir(args.audio_dir):
        sys.exit(f"Error: '{args.audio_dir}' is not a valid directory")

    if DEBUG_VRAM and not torch.cuda.is_available():
        print("⚠️  --debug-vram was enabled but CUDA is not available; continuing without VRAM metrics.")

    print(f"Loading model from {args.model_path} …")
    model, processor = load_model(args.model_path)

    system_prompt = args.system_prompt or SYSTEM_PROMPT
    extensions = {".wav", ".mp3", ".flac", ".aac", ".ogg", ".m4a"}

    for fname in os.listdir(args.audio_dir):
        base, ext = os.path.splitext(fname)
        if ext.lower() not in extensions:
            continue

        audio_path = os.path.join(args.audio_dir, fname)
        print(f"\n▶︎ Analyzing: {audio_path}")
        if DEBUG_VRAM:
            log_vram("before analyze_audio() call")

        result_json = analyze_audio(audio_path, model, processor, system_prompt=system_prompt)

        output_path = os.path.join(args.audio_dir, f"{base}_analysis.json")
        try:
            with open(output_path, "w", encoding="utf-8") as fp:
                fp.write(str(result_json))
            print(f"✔ Saved → {output_path}")
        except Exception as e:
            print(f"✖ Failed saving {fname}: {e}")

    if DEBUG_VRAM:
        log_vram("all files processed — exiting")


if __name__ == "__main__":  # pragma: no cover
    main()
