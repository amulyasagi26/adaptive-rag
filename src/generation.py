"""Lazy Qwen answer generation over retrieved chunks."""

from typing import Any, Sequence


def build_generation_prompt(question: str, retrieved_chunks: Sequence[Any]) -> list[dict[str, str]]:
    """Build a grounded chat prompt without loading a language model."""
    context = "\n\n".join(
        f"Document {index} ({chunk.title}):\n{chunk.text}"
        for index, item in enumerate(retrieved_chunks, start=1)
        for chunk in [item["chunk"] if isinstance(item, dict) else item]
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Use only the provided context to answer "
                "the question briefly and accurately. If the answer is not in the "
                "context, say: I cannot answer this based on the provided context."
            ),
        },
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]


class QwenGenerator:
    """Lazy wrapper around a Qwen instruction model.

    Constructing this class downloads/loads the model; importing this module
    does not.  The default generation is deterministic for reproducible
    evaluation, while sampling can be enabled explicitly.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str | None = None,
        torch_dtype: Any | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install torch and transformers to use QwenGenerator") from exc

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        load_kwargs: dict[str, Any] = {}
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        if device == "auto":
            load_kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if device and device != "auto":
            self.model.to(device)
        self.device = device or str(next(self.model.parameters()).device)

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: Sequence[Any],
        max_new_tokens: int = 150,
        do_sample: bool = False,
        temperature: float = 0.3,
    ) -> str:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        messages = build_generation_prompt(question, retrieved_chunks)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
        generated = self.model.generate(**model_inputs, **generation_kwargs)
        new_tokens = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated)
        ]
        return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
