import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from snac import SNAC
import soundfile as sf

# --------------- Performance knobs ---------------
USE_TORCH_COMPILE = True          # Set False if your env/PyTorch < 2.1 or errors out
USE_MEMORY_EFFICIENT_ATTN = True  # Will be enabled if the model supports it (xFormers/FlashAttention)
DTYPE_FOR_COMPUTE = torch.bfloat16  # good default on RTX 40-series; try torch.float16 if needed

# --------------- Torch backend hints ---------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")  # helps on Ampere/Lovelace

# --------------- Quantization config (4-bit) ---------------
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=DTYPE_FOR_COMPUTE,
    bnb_4bit_use_double_quant=True,
)

# --------------- Control tokens (fixed for Veena) ---------------
START_OF_SPEECH_TOKEN = 128257
END_OF_SPEECH_TOKEN   = 128258
START_OF_HUMAN_TOKEN  = 128259
END_OF_HUMAN_TOKEN    = 128260
START_OF_AI_TOKEN     = 128261
END_OF_AI_TOKEN       = 128262
AUDIO_CODE_BASE_OFFSET = 128266  # 7 codebooks, each size 4096 (indices 0..4095)

# --------------- Load model/tokenizer ---------------
t0 = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(
    "veena-tts",
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("veena-tts", trust_remote_code=True)

# Some models don’t ship pad_token_id; use eos as pad to keep generate() happy.
if tokenizer.pad_token_id is None:
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "</pad>"})
    tokenizer.pad_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id

# Try memory-efficient attention (if supported)
if USE_MEMORY_EFFICIENT_ATTN:
    try:
        # Many custom models expose a helper; if not, ignore.
        if hasattr(model, "set_use_memory_efficient_attention_xformers"):
            model.set_use_memory_efficient_attention_xformers(True)
    except Exception:
        pass

# Optional: torch.compile to reduce Python overhead in generate()
if USE_TORCH_COMPILE:
    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    except Exception:
        # Fallback silently if unsupported
        pass

model.eval()
t1 = time.perf_counter()
print(f"[Load LLM] {t1 - t0:.3f}s")

# --------------- Load SNAC decoder ---------------
t2 = time.perf_counter()
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()
t3 = time.perf_counter()
print(f"[Load SNAC] {t3 - t2:.3f}s")

# --------------- Vectorized SNAC helpers ---------------
@torch.inference_mode()
def filter_snac_tokens_vectorized(generated_ids: torch.Tensor) -> torch.Tensor:
    """
    generated_ids: 1D tensor on GPU
    Returns: filtered 1D tensor (SNAC tokens only) on GPU
    """
    # Keep tokens in range [BASE, BASE + 7*4096)
    upper = AUDIO_CODE_BASE_OFFSET + 7 * 4096
    mask = (generated_ids >= AUDIO_CODE_BASE_OFFSET) & (generated_ids < upper)
    return generated_ids[mask]

@torch.inference_mode()
def deinterleave_snac_tokens_vectorized(snac_tokens: torch.Tensor):
    """
    snac_tokens: 1D tensor length multiple of 7, on GPU
    Returns: list of 3 tensors [level0, level1, level2] on GPU (int32), batch dim added
    Level 0: 1 token per frame
    Level 1: 2 tokens per frame (columns 1 and 4)
    Level 2: 4 tokens per frame (columns 2,3,5,6)
    """
    if snac_tokens.numel() == 0 or (snac_tokens.numel() % 7 != 0):
        return None

    frames = snac_tokens.view(-1, 7)  # shape (T, 7)

    # Offsets per codebook (size 7)
    offsets = torch.arange(7, device=snac_tokens.device, dtype=snac_tokens.dtype) * 4096 + AUDIO_CODE_BASE_OFFSET
    # Subtract offsets per column (broadcast)
    frames = frames - offsets  # (T,7), now each entry should be in [0..4095]

    # Sanity clamp/check (optional safety)
    if torch.any((frames < 0) | (frames > 4095)):
        raise ValueError("Invalid SNAC token values after offset subtraction.")

    # Extract levels
    lvl0 = frames[:, 0]  # (T,)
    lvl1 = torch.stack((frames[:, 1], frames[:, 4]), dim=1).reshape(-1)  # (2T,)
    lvl2 = torch.stack((frames[:, 2], frames[:, 3], frames[:, 5], frames[:, 6]), dim=1).reshape(-1)  # (4T,)

    # SNAC expects int32 with batch dim (1, L)
    lvl0 = lvl0.to(torch.int32).unsqueeze(0)
    lvl1 = lvl1.to(torch.int32).unsqueeze(0)
    lvl2 = lvl2.to(torch.int32).unsqueeze(0)
    return [lvl0, lvl1, lvl2]

# --------------- Build input IDs helper (GPU-first) ---------------
def build_input_ids(prompt_text: str, speaker: str, device) -> torch.Tensor:
    # Compose prompt with speaker tag on CPU string, then tokenize -> tensor, send to GPU once.
    prompt = f"<spk_{speaker}> {prompt_text}"
    tok = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    prompt_ids = tok["input_ids"].to(device, non_blocking=True)  # shape (1, L)

    # Prepend/append control tokens (build them directly on GPU)
    control = torch.tensor(
        [[START_OF_HUMAN_TOKEN]], device=device, dtype=prompt_ids.dtype
    )
    control_end = torch.tensor(
        [[END_OF_HUMAN_TOKEN, START_OF_AI_TOKEN, START_OF_SPEECH_TOKEN]],
        device=device,
        dtype=prompt_ids.dtype,
    )

    input_ids = torch.cat([control, prompt_ids, control_end], dim=1)  # shape (1, L+control)
    return input_ids

# --------------- Core TTS function ---------------
@torch.inference_mode()
def generate_speech(
    text: str,
    speaker: str = "kavya",
    temperature: float = 0.4,
    top_p: float = 0.9,
    max_new_tokens: int | None = None,
):
    device = next(model.parameters()).device

    t_build0 = time.perf_counter()
    input_ids = build_input_ids(text, speaker, device)
    t_build1 = time.perf_counter()

    # Heuristic: Veena emits 7 SNAC tokens per 'frame'; this keeps a sane upper bound.
    if max_new_tokens is None:
        approx = max(64, min(int(len(text) * 1.3) * 7 + 21, 700))
        max_new_tokens = approx

    # Generate (GPU-bound; Python overhead minimized)
    t_gen0 = time.perf_counter()
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=[END_OF_SPEECH_TOKEN, END_OF_AI_TOKEN],
    )
    t_gen1 = time.perf_counter()

    # Slice to only the newly generated ids
    new_tokens = output[0, input_ids.shape[1]:]

    # Vectorized SNAC filtering on GPU
    t_filt0 = time.perf_counter()
    snac_tokens = filter_snac_tokens_vectorized(new_tokens)
    t_filt1 = time.perf_counter()

    if snac_tokens.numel() == 0:
        raise ValueError("No audio tokens generated.")

    # Vectorized de-interleaving on GPU
    t_deint0 = time.perf_counter()
    hierarchical_codes = deinterleave_snac_tokens_vectorized(snac_tokens)
    t_deint1 = time.perf_counter()
    if hierarchical_codes is None:
        raise ValueError("SNAC tokens not a multiple of 7 or invalid shape.")

    # SNAC decode (GPU) -> returns audio on GPU (float)
    t_dec0 = time.perf_counter()
    audio_hat = snac_model.decode(hierarchical_codes)  # shape (1, n)
    # Post-process to CPU numpy
    audio = audio_hat.squeeze().clamp(-1, 1).detach().to("cpu").numpy()
    t_dec1 = time.perf_counter()

    # Timing breakdown
    timings = {
        "build_input": t_build1 - t_build0,
        "generate": t_gen1 - t_gen0,
        "filter_snac": t_filt1 - t_filt0,
        "deinterleave": t_deint1 - t_deint0,
        "snac_decode": t_dec1 - t_dec0,
        "total": t_dec1 - t_build0,
    }

    return audio, timings

# --------------- Example usage / sanity test ---------------
def run_examples():
    tests = [
#        ("आज मैंने एक नई तकनीक के बारे में सीखा जो कृत्रिम बुद्धिमत्ता का उपयोग करके मानव जैसी आवाज़ उत्पन्न कर सकती है।", "kavya", "output_hindi_kavya.wav"),
        ("Welcome to cultyvate... This is a verification call... Please press one if you are Lucky Singh Son of Haripreeth Singh. If not, press two.", "agastya", "agastya.wav"),
#        ("मैं तो पूरा presentation prepare कर चुका हूं! कल रात को ही मैंने पूरा code base चेक किया।", "maitri", "output_mixed_maitri.wav"),
    ]

    for text, spk, fname in tests:
        t0 = time.perf_counter()
        audio, t = generate_speech(text, speaker=spk)
        t1 = time.perf_counter()
        sf.write(fname, audio, 24000)
        print(f"\n[{fname}] wrote {len(audio)/24000:.2f}s of audio")
        print(f"Timings: {t}")
        print(f"End-to-end (incl. file write): {t1 - t0:.3f}s")

if __name__ == "__main__":
    run_examples()

