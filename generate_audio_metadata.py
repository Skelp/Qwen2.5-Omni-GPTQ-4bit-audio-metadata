"""
Audio Metadata Generator with Qwen2.5-Omni GPTQ Model

Additional, Non-Optional Requirements:
- flash_attn
- qwen_omni_utils

Usable Environment Example:
- Arch Linux
- RTX 4070 Ti SUPER (16 GB VRAM)
- Python 3.10.17
- Reference `example_environment.txt` for installed packages

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 ./generate_audio_metadata.py '{your_audio_directory}' --model-path Qwen/Qwen2.5-Omni-7B-GPTQ-Int4

Expected VRAM Usage: 
    average of 8 GB, spikes near the start between 12-16 GB (optimization needed still)
"""

import os
import torch
import argparse
from typing import Dict, Any

# GPTQ imports

# shoutout to the Qwen Team for making the sub-directories contain hyphens! :)
# XOXO
import os, importlib.util, sys

here = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(here, "low-VRAM-mode", "modeling_qwen2_5_omni_low_VRAM_mode.py")

spec = importlib.util.spec_from_file_location("modeling_qwen2_5_omni_low_VRAM_mode", module_path)
module = importlib.util.module_from_spec(spec)

spec.loader.exec_module(module)

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
        "token2wav", 
        "thinker.audio_tower", 
        "thinker.model.rotary_emb",
        "thinker.visual", 
        "talker"
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
        self.thinker.visual = move_to(self.thinker.visual, device=self.quantize_config.device)
        self.thinker.audio_tower = move_to(self.thinker.audio_tower, device=self.quantize_config.device)

    def pre_quantize_generate_hook_end(self):
        self.thinker.visual = move_to(self.thinker.visual, device=CPU)
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
    
    # Store model_path from kwargs before creating model
    model_path = kwargs.get("model_path", None)
    
    model = cls._from_config(config, **kwargs)
    
    if model_path:
        spk_path = cached_file(
            model_path,
            "spk_dict.pt",
            subfolder=kwargs.pop("subfolder", None),
            cache_dir=kwargs.pop("cache_dir", None),
            force_download=kwargs.pop("force_download", False),
            proxies=kwargs.pop("proxies", None),
            resume_download=kwargs.pop("resume_download", None),
            local_files_only=kwargs.pop("local_files_only", False),
            token=kwargs.pop("use_auth_token", None),
            revision=kwargs.pop("revision", None),
        )
        if spk_path is None:
            raise ValueError(f"Speaker dictionary not found at {spk_path}")
        
        model.load_speakers(spk_path)
    
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
"""

def load_model(model_path, device_map=None):
    """Load the GPTQ model and processor"""
    # Download model if using HuggingFace repo
    if not os.path.exists(model_path):
        model_path = snapshot_download(repo_id=model_path)
    
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
        model_path=model_path  # Pass model_path for speaker dictionary loading
    )
    
    # Disable audio output components since we only need text
    try:
        if hasattr(model, 'disable_talker'):
            model.disable_talker()
            print("Audio output components disabled")
        if hasattr(model, 'has_talker'):
            model.has_talker = False
    except Exception as e:
        print(f"Note: Could not disable audio components: {e}")
    
    # Ensure visual and audio_tower are on CPU to save VRAM
    if hasattr(model, 'thinker'):
        if hasattr(model.thinker, 'visual') and model.thinker.visual is not None:
            model.thinker.visual = model.thinker.visual.to('cpu')
        if hasattr(model.thinker, 'audio_tower') and model.thinker.audio_tower is not None:
            model.thinker.audio_tower = model.thinker.audio_tower.to('cpu')
    
    # Clear any residual GPU memory
    torch.cuda.empty_cache()
    
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    
    return model, processor

def analyze_audio(audio_path, model, processor, system_prompt=SYSTEM_PROMPT):
    """Analyze an audio file and return the text response"""
    
    # Prepare messages
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "audio", "audio": audio_path}]}
    ]
    
    # Apply chat template
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    # Process multimedia info
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    
    # Prepare inputs
    inputs = processor(text=text, audio=audios, images=images, videos=videos, 
                      return_tensors="pt", padding=True, use_audio_in_video=False)
    
    # Move visual and audio components to GPU only when needed
    if hasattr(model, 'thinker'):
        if hasattr(model.thinker, 'visual') and model.thinker.visual is not None:
            model.thinker.visual = model.thinker.visual.to('cuda')
        if hasattr(model.thinker, 'audio_tower') and model.thinker.audio_tower is not None:
            model.thinker.audio_tower = model.thinker.audio_tower.to('cuda')
    
    # Move inputs to GPU with proper dtype handling
    inputs_cuda = {}
    for k, v in inputs.items():
        if hasattr(v, 'to'):
            inputs_cuda[k] = v.to('cuda')
            if hasattr(model, 'dtype') and hasattr(v, 'dtype') and v.dtype.is_floating_point:
                inputs_cuda[k] = inputs_cuda[k].to(model.dtype)
        else:
            inputs_cuda[k] = v
    
    # Generate response using the thinker component directly (text only)
    with torch.no_grad():
        # Use the thinker model directly to avoid speaker validation
        if hasattr(model, 'thinker'):
            output_ids = model.thinker.generate(
                **inputs_cuda, 
                max_new_tokens=4096,
                use_audio_in_video=False
            )
        else:
            # Fallback to regular generate with explicit parameters
            output_ids = model.generate(
                **inputs_cuda, 
                return_audio=False,
                use_audio_in_video=False
            )
    
    # Move visual and audio components back to CPU to save VRAM
    if hasattr(model, 'thinker'):
        if hasattr(model.thinker, 'visual') and model.thinker.visual is not None:
            model.thinker.visual = model.thinker.visual.to('cpu')
        if hasattr(model.thinker, 'audio_tower') and model.thinker.audio_tower is not None:
            model.thinker.audio_tower = model.thinker.audio_tower.to('cpu')
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    generate_ids = output_ids[:, inputs.input_ids.size(1):]

    # Decode the response
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    return response

def main():
    parser = argparse.ArgumentParser(
        description="Analyze a directory of audio files using Qwen2.5-Omni GPTQ model"
    )
    parser.add_argument(
        "audio_dir", type=str,
        help="Path to the directory containing audio files to analyze"
    )
    parser.add_argument(
        "--model-path", type=str,
        default="Qwen/Qwen2.5-Omni-7B-GPTQ-Int4",
        help="Path to the GPTQ model (local or HuggingFace repo)"
    )
    parser.add_argument(
        "--system-prompt", type=str, default=None,
        help="Custom system prompt (optional)"
    )
    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.audio_dir):
        print(f"Error: Directory '{args.audio_dir}' not found or is not a directory!")
        return

    print(f"Loading model from {args.model_path}...")
    model, processor = load_model(args.model_path)

    # Determine system prompt
    system_prompt = args.system_prompt if args.system_prompt else SYSTEM_PROMPT

    # Supported audio extensions
    extensions = {'.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a'}

    # Process each file in the directory
    for fname in os.listdir(args.audio_dir):
        base, ext = os.path.splitext(fname)
        if ext.lower() not in extensions:
            continue

        audio_path = os.path.join(args.audio_dir, fname)
        print(f"\nAnalyzing audio file: {audio_path}")
        print("Note: You may see a warning about system prompt - this is normal and can be ignored.")

        # Run analysis
        response = analyze_audio(audio_path, model, processor, system_prompt)

        # Prepare output JSON path
        output_filename = f"{base}_analysis.json"
        output_path = os.path.join(args.audio_dir, output_filename)

        # Save results as JSON
        try:
            with open(output_path, 'w', encoding='utf-8') as out_f:
                # If response is not already a dict, wrap it
                out_f.write(str(response))
            print(f"Analysis saved to: {output_path}")
        except Exception as e:
            print(f"Failed to save analysis for {fname}: {e}")

if __name__ == "__main__":
    main()