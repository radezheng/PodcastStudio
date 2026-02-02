from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VibeVoiceEngineConfig:
    model_id: str
    device: str
    cfg_scale: float
    disable_prefill: bool
    ddpm_steps: int


_ENGINE_LOCK = threading.Lock()
_ENGINE: "VibeVoiceEngine | None" = None

_LOG = logging.getLogger("podcaststudio.vibevoice")
_LOG.setLevel(logging.INFO)


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _str_env(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None:
        return default
    val = raw.strip().strip('"')
    return val if val else default


def _pick_device() -> str:
    # Import torch lazily so local dev without torch can still import this module.
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def load_engine_config() -> VibeVoiceEngineConfig:
    model_id = os.environ.get("VIBEVOICE_MODEL_ID", "vibevoice/VibeVoice-1.5B").strip().strip('"')
    device = os.environ.get("VIBEVOICE_DEVICE", "").strip().strip('"') or _pick_device()
    cfg_scale = _float_env("VIBEVOICE_CFG_SCALE", 1.3)
    disable_prefill = _bool_env("VIBEVOICE_DISABLE_PREFILL", False)
    ddpm_steps = _int_env("VIBEVOICE_DDPM_STEPS", 10)
    return VibeVoiceEngineConfig(
        model_id=model_id,
        device=device,
        cfg_scale=cfg_scale,
        disable_prefill=disable_prefill,
        ddpm_steps=ddpm_steps,
    )


class VibeVoiceEngine:
    def __init__(self, cfg: VibeVoiceEngineConfig):
        self.cfg = cfg
        self._processor = None
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._processor is not None and self._model is not None

    def _ensure_loaded(self) -> None:
        if self.is_loaded:
            return

        import torch  # type: ignore

        from vibevoice.modular.modeling_vibevoice_inference import (  # type: ignore
            VibeVoiceForConditionalGenerationInference,
        )
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor  # type: ignore

        cfg = self.cfg

        processor = VibeVoiceProcessor.from_pretrained(cfg.model_id)

        if cfg.device == "mps":
            load_dtype = torch.float32
            attn_impl = _str_env("VIBEVOICE_ATTN_IMPL", "sdpa")
        elif cfg.device == "cuda":
            # T4 (compute capability 7.5) does NOT support BF16.
            # Using BF16 can force fallbacks/extra memory and cause OOM.
            try:
                bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported")())  # type: ignore[misc]
            except Exception:
                try:
                    major, _minor = torch.cuda.get_device_capability()  # type: ignore[misc]
                    bf16_ok = major >= 8
                except Exception:
                    bf16_ok = False

            load_dtype = torch.bfloat16 if bf16_ok else torch.float16
            # Default to SDPA to avoid optional flash-attn builds on ACA.
            attn_impl = _str_env("VIBEVOICE_ATTN_IMPL", "sdpa")
        else:
            load_dtype = torch.float32
            attn_impl = _str_env("VIBEVOICE_ATTN_IMPL", "sdpa")

        use_8bit = bool(cfg.device == "cuda" and _bool_env("VIBEVOICE_LOAD_IN_8BIT", False))
        quantization_config = None
        if use_8bit:
            # Reduce VRAM pressure (especially on T4) by loading weights in 8-bit.
            # Requires bitsandbytes.
            try:
                from transformers import BitsAndBytesConfig  # type: ignore

                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            except Exception:
                # If bitsandbytes/transformers bitsandbytes integration is unavailable, fall back.
                quantization_config = None

        _LOG.info(
            "Loading VibeVoice model. device=%s torch_dtype=%s attn_implementation=%s 8bit=%s model_id=%s",
            cfg.device,
            str(load_dtype),
            attn_impl,
            bool(quantization_config is not None),
            cfg.model_id,
        )

        load_kwargs = {
            "torch_dtype": load_dtype,
            "attn_implementation": attn_impl,
            "low_cpu_mem_usage": True,
        }
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config

        try:
            if cfg.device == "mps":
                model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    cfg.model_id,
                    device_map=None,
                    **load_kwargs,
                )
                model.to("mps")
            elif cfg.device == "cuda":
                model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    cfg.model_id,
                    device_map={"": 0},
                    **load_kwargs,
                )
            else:
                model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    cfg.model_id,
                    device_map="cpu",
                    **load_kwargs,
                )
        except Exception:
            # SDPA is the most broadly compatible fallback.
            _LOG.exception("Primary model load failed; retrying with attn_implementation=sdpa")
            load_kwargs["attn_implementation"] = "sdpa"

            device_map = None
            if cfg.device == "cuda":
                device_map = {"": 0}
            elif cfg.device == "cpu":
                device_map = "cpu"
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                cfg.model_id,
                device_map=device_map,
                **load_kwargs,
            )
            if cfg.device == "mps":
                model.to("mps")

        model.eval()
        model.set_ddpm_inference_steps(num_steps=cfg.ddpm_steps)

        self._processor = processor
        self._model = model

    def synthesize(
        self,
        *,
        text: str,
        voice_samples: list[str],
        output_path: Path,
        seed: int | None = None,
    ) -> None:
        self._ensure_loaded()

        import torch  # type: ignore

        assert self._processor is not None
        assert self._model is not None

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Prepare inputs for the model.
        inputs = self._processor(
            text=[text],
            voice_samples=[voice_samples],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        target_device = self.cfg.device if self.cfg.device != "cpu" else "cpu"
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(target_device)

        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=self.cfg.cfg_scale,
                tokenizer=self._processor.tokenizer,
                generation_config={"do_sample": False},
                verbose=False,
                is_prefill=not self.cfg.disable_prefill,
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._processor.save_audio(outputs.speech_outputs[0], output_path=str(output_path))


def get_engine() -> VibeVoiceEngine:
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = VibeVoiceEngine(load_engine_config())
        return _ENGINE


def start_background_warmup() -> None:
    if not _bool_env("VIBEVOICE_WARMUP", True):
        return

    def _run() -> None:
        try:
            _LOG.info("VibeVoice warmup started")
            get_engine()._ensure_loaded()
            _LOG.info("VibeVoice warmup complete")
        except Exception:
            # Warmup failures should not crash the worker;
            # the job execution path will surface the real error.
            _LOG.exception("VibeVoice warmup failed")
            return

    threading.Thread(target=_run, name="vibevoice-warmup", daemon=True).start()
