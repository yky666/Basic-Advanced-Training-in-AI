"""DeepSeek-VL image description helper.

This module is intentionally independent from YOLO code. It lazily loads the
DeepSeek-VL model only when semantic description is requested.

Typical usage:
    describer = DeepSeekVLDescriber()
    text = describer.describe_image("crop.jpg")

Notes:
    - Do not call this on every video frame. It is much slower than YOLO.
    - Recommended trigger: stable detection, high confidence, manual hotkey,
      or a cooldown timer.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DEEPSEEK_REPO = PROJECT_ROOT.parent / "voice_result" / "DeepSeek-VL"
DEFAULT_MODEL_PATH = Path(r"D:\ai_models\deepseek-vl-1.3b-base")
DEFAULT_PROMPT = (
    "Describe the cropped image briefly. "
    "Mention the main object, its visual attributes, and any useful context."
)


def clean_deepseek_text(text: str) -> str:
    """Clean byte-level tokenizer artifacts from decoded DeepSeek-VL text."""
    replacements = {
        chr(0x0120): " ",   # GPT-2 byte-level space marker
        chr(0x010A): "\n", # GPT-2 byte-level newline marker
        chr(0x0109): "\t", # GPT-2 byte-level tab marker
        chr(0x00C2): "",    # stray Latin-1 marker sometimes seen on Windows
    }
    for old, new_value in replacements.items():
        text = text.replace(old, new_value)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


class DeepSeekVLDescriber:
    """Lazy DeepSeek-VL wrapper for single-image descriptions."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        repo_path: str | Path = DEFAULT_DEEPSEEK_REPO,
        max_new_tokens: int = 96,
    ) -> None:
        self.model_path = Path(model_path)
        self.repo_path = Path(repo_path)
        self.max_new_tokens = max_new_tokens
        self.tokenizer = None
        self.processor = None
        self.model = None

    def load(self) -> None:
        """Load DeepSeek-VL once; later calls reuse the loaded model."""
        if self.model is not None:
            return
        if not self.repo_path.exists():
            raise FileNotFoundError(f"DeepSeek-VL repo not found: {self.repo_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"DeepSeek-VL model not found: {self.model_path}")
        repo_str = str(self.repo_path)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        from deepseek_vl.utils.io import load_pretrained_model

        self.tokenizer, self.processor, self.model = load_pretrained_model(str(self.model_path))

    def describe_image(
        self,
        image: str | Path | Image.Image,
        prompt: str = DEFAULT_PROMPT,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Describe one image path or PIL image with DeepSeek-VL."""
        self.load()
        assert self.tokenizer is not None
        assert self.processor is not None
        assert self.model is not None

        if isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            pil_image = Image.open(image).convert("RGB")

        conversation = [
            {
                "role": "User",
                "content": f"<image_placeholder>{prompt}",
                "images": ["<in_memory_image>"],
            },
            {"role": "Assistant", "content": ""},
        ]
        inputs = self.processor(
            conversations=conversation,
            images=[pil_image],
            force_batchify=True,
        ).to(self.model.device)
        inputs_embeds = self.model.prepare_inputs_embeds(**inputs)
        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        decoded = self.tokenizer.decode(outputs[0].detach().cpu().tolist(), skip_special_tokens=True)
        return clean_deepseek_text(decoded)


def describe_image_once(
    image: str | Path | Image.Image,
    prompt: str = DEFAULT_PROMPT,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    repo_path: str | Path = DEFAULT_DEEPSEEK_REPO,
) -> str:
    """Convenience one-shot API. Prefer DeepSeekVLDescriber for repeated calls."""
    return DeepSeekVLDescriber(model_path=model_path, repo_path=repo_path).describe_image(image, prompt=prompt)
