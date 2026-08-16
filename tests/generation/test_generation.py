"""Tests for autoregressive generation.

Most of these use a *scripted* model rather than a trained one. A randomly
initialized Transformer emits essentially uniform logits, which cannot
distinguish greedy from sampling, cannot be made to emit EOS on demand, and
cannot show that the context was cropped correctly. A model whose logits are
dictated by the test can do all three.

The real Transformer is still exercised, for the properties that need it:
shapes, dtypes, determinism, and that generation runs past the 2048-token
context boundary without raising.
"""

from __future__ import annotations

import pytest
import torch

from llm.generation import (
    GenerationResult,
    SamplingConfig,
    StreamingDecoder,
    apply_top_k,
    apply_top_p,
    decode_generated,
    generate,
    generate_ids,
    select_next_token,
    undecodable_token_ids,
)
from llm.model import ModelConfig
from llm.model.transformer import Transformer
from llm.tokenizer import Tokenizer


class ScriptedModel(torch.nn.Module):
    """Emits logits chosen by the test, and records what it was fed."""

    def __init__(self, config: ModelConfig, favoured: int | None = None) -> None:
        super().__init__()
        self.config = config
        self.favoured = favoured
        self.calls: list[list[int]] = []
        self._parameter = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.calls.append(input_ids[0].tolist())
        batch, sequence = input_ids.shape
        logits = torch.zeros(batch, sequence, self.config.vocab_size)
        if self.favoured is not None:
            logits[..., self.favoured] = 100.0
        return logits


def small_config(**overrides) -> ModelConfig:
    base = dict(
        vocab_size=64,
        context_length=16,
        n_layers=2,
        hidden_size=16,
        n_heads=4,
        head_dim=4,
        ffn_hidden_size=32,
    )
    base.update(overrides)
    return ModelConfig(**base)


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    corpus = [
        "The purpose of a voltage regulator is to hold output voltage steady.",
        "In Python, a dictionary maps keys to values.",
        "Newton's second law states that force equals mass times acceleration.",
        "A database transaction should be atomic, consistent, isolated, durable.",
    ]
    return Tokenizer.train(corpus, vocab_size=512, min_pair_frequency=1)


@pytest.fixture(autouse=True)
def deterministic_seed():
    torch.manual_seed(20260816)


# --------------------------------------------------------------------------
# sampling configuration
# --------------------------------------------------------------------------


def test_defaults_are_sane():
    config = SamplingConfig()
    assert config.max_new_tokens > 0
    assert config.temperature > 0
    assert config.add_bos is True
    assert config.stop_on_eos is True


@pytest.mark.parametrize(
    "overrides",
    [
        dict(max_new_tokens=0),
        dict(max_new_tokens=-1),
        dict(temperature=-0.5),
        dict(top_k=0),
        dict(top_k=-3),
        dict(top_p=0.0),
        dict(top_p=1.5),
        dict(max_new_tokens=1.5),
        dict(top_k=2.5),
    ],
)
def test_invalid_sampling_configs_are_rejected(overrides):
    with pytest.raises((ValueError, TypeError)):
        SamplingConfig(**overrides)


def test_greedy_is_recognized_two_ways():
    assert SamplingConfig(temperature=0.0).is_greedy
    assert SamplingConfig(top_k=1).is_greedy
    assert not SamplingConfig(temperature=0.8, top_k=40).is_greedy


# --------------------------------------------------------------------------
# logit filters
# --------------------------------------------------------------------------


def test_top_k_keeps_exactly_k_candidates():
    logits = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0])
    filtered = apply_top_k(logits, 2)

    assert torch.isfinite(filtered).sum() == 2
    assert filtered[1].isfinite() and filtered[4].isfinite()


def test_top_k_larger_than_vocab_is_harmless():
    logits = torch.tensor([1.0, 2.0, 3.0])
    assert torch.isfinite(apply_top_k(logits, 99)).all()


def test_top_p_keeps_the_smallest_sufficient_set():
    # softmax over these is dominated by the last entry
    logits = torch.tensor([0.0, 0.0, 10.0])
    filtered = apply_top_p(logits, 0.9)

    assert filtered[2].isfinite()
    assert torch.isfinite(filtered).sum() == 1


def test_top_p_always_keeps_at_least_one_token():
    """A single token above the threshold must not empty the distribution."""

    logits = torch.tensor([0.0, 0.0, 50.0])
    filtered = apply_top_p(logits, 0.1)

    assert torch.isfinite(filtered).sum() >= 1
    probabilities = torch.softmax(filtered, dim=-1)
    assert torch.isfinite(probabilities).all()
    assert probabilities.sum() == pytest.approx(1.0)


def test_top_p_preserves_position():
    """Filtering must not reorder the vocabulary."""

    logits = torch.tensor([1.0, 9.0, 2.0, 8.0])
    filtered = apply_top_p(logits, 0.99)

    kept = torch.isfinite(filtered)
    assert filtered[kept].tolist() == [v for v, k in zip(logits.tolist(), kept) if k]


# --------------------------------------------------------------------------
# token selection
# --------------------------------------------------------------------------


def test_zero_temperature_is_argmax():
    logits = torch.tensor([0.1, 0.9, 0.4])
    assert select_next_token(logits, SamplingConfig(temperature=0.0)) == 1


def test_top_k_one_is_argmax():
    logits = torch.tensor([0.1, 0.9, 0.4])
    config = SamplingConfig(temperature=1.0, top_k=1)
    assert all(select_next_token(logits, config) == 1 for _ in range(20))


def test_sampling_is_reproducible_under_a_seed():
    logits = torch.randn(50)
    config = SamplingConfig(temperature=1.0, top_k=None, seed=7)

    first = [
        select_next_token(logits, config, torch.Generator().manual_seed(7))
        for _ in range(5)
    ]
    second = [
        select_next_token(logits, config, torch.Generator().manual_seed(7))
        for _ in range(5)
    ]
    assert first == second


def test_higher_temperature_produces_more_variety():
    logits = torch.tensor([3.0, 2.5, 2.0, 1.5, 1.0])

    cold = {
        select_next_token(
            logits, SamplingConfig(temperature=0.1, top_k=None),
            torch.Generator().manual_seed(i),
        )
        for i in range(40)
    }
    hot = {
        select_next_token(
            logits, SamplingConfig(temperature=5.0, top_k=None),
            torch.Generator().manual_seed(i),
        )
        for i in range(40)
    }
    assert len(hot) > len(cold)


def test_non_vector_logits_are_rejected():
    with pytest.raises(ValueError):
        select_next_token(torch.randn(2, 5), SamplingConfig())


# --------------------------------------------------------------------------
# the generation loop, with a scripted model
# --------------------------------------------------------------------------


def test_generates_exactly_max_new_tokens():
    model = ScriptedModel(small_config(), favoured=5)
    produced, stopped = generate_ids(
        model, [1, 2, 3], SamplingConfig(max_new_tokens=7, temperature=0.0)
    )

    assert len(produced) == 7
    assert produced == [5] * 7
    assert stopped is False


def test_stops_on_eos_and_excludes_it():
    model = ScriptedModel(small_config(), favoured=9)
    produced, stopped = generate_ids(
        model,
        [1, 2],
        SamplingConfig(max_new_tokens=10, temperature=0.0),
        eos_id=9,
    )

    assert produced == []
    assert stopped is True


def test_eos_can_be_ignored():
    model = ScriptedModel(small_config(), favoured=9)
    produced, stopped = generate_ids(
        model,
        [1, 2],
        SamplingConfig(max_new_tokens=4, temperature=0.0, stop_on_eos=False),
        eos_id=9,
    )

    assert produced == [9, 9, 9, 9]
    assert stopped is False


def test_the_prompt_is_fed_back_with_each_new_token():
    model = ScriptedModel(small_config(), favoured=5)
    generate_ids(model, [1, 2, 3], SamplingConfig(max_new_tokens=3, temperature=0.0))

    assert model.calls[0] == [1, 2, 3]
    assert model.calls[1] == [1, 2, 3, 5]
    assert model.calls[2] == [1, 2, 3, 5, 5]


def test_context_is_cropped_from_the_left():
    """Never feed the model more positions than it has RoPE tables for."""

    config = small_config(context_length=8)
    model = ScriptedModel(config, favoured=5)

    generate_ids(
        model, list(range(1, 9)), SamplingConfig(max_new_tokens=4, temperature=0.0)
    )

    assert all(len(call) <= 8 for call in model.calls), [len(c) for c in model.calls]
    # The final call happens before the last token is appended, so the window
    # holds four prompt tokens and the three generated so far.
    assert model.calls[-1] == [4, 5, 6, 7, 8, 5, 5, 5]


def test_a_prompt_longer_than_the_context_is_cropped():
    config = small_config(context_length=8)
    model = ScriptedModel(config, favoured=5)

    generate_ids(
        model, list(range(1, 21)), SamplingConfig(max_new_tokens=2, temperature=0.0)
    )

    assert len(model.calls[0]) == 8
    assert model.calls[0] == [13, 14, 15, 16, 17, 18, 19, 20]


def test_empty_prompt_is_rejected():
    model = ScriptedModel(small_config())
    with pytest.raises(ValueError, match="must not be empty"):
        generate_ids(model, [], SamplingConfig())


def test_on_token_callback_receives_each_token():
    model = ScriptedModel(small_config(), favoured=5)
    seen: list[int] = []

    generate_ids(
        model,
        [1],
        SamplingConfig(max_new_tokens=3, temperature=0.0),
        on_token=seen.append,
    )
    assert seen == [5, 5, 5]


def test_forbidden_ids_are_never_sampled():
    """The scripted model wants token 5; masking it must force something else."""

    model = ScriptedModel(small_config(), favoured=5)
    produced, _ = generate_ids(
        model,
        [1],
        SamplingConfig(max_new_tokens=6, temperature=0.0),
        forbidden_ids=[5],
    )

    assert produced
    assert 5 not in produced


def test_masking_everything_is_reported_rather_than_sampled():
    config = small_config()
    model = ScriptedModel(config)

    with pytest.raises(ValueError, match="every token id was masked"):
        generate_ids(
            model,
            [1],
            SamplingConfig(max_new_tokens=1, temperature=0.0),
            forbidden_ids=list(range(config.vocab_size)),
        )


def test_training_mode_is_restored():
    model = ScriptedModel(small_config(), favoured=5)
    model.train()
    generate_ids(model, [1], SamplingConfig(max_new_tokens=1, temperature=0.0))
    assert model.training


# --------------------------------------------------------------------------
# end to end, with the real model and tokenizer
# --------------------------------------------------------------------------


def test_generate_returns_a_populated_result(tokenizer):
    model = Transformer(small_config(vocab_size=tokenizer.vocab_size))
    result = generate(
        model, tokenizer, "The purpose of", SamplingConfig(max_new_tokens=5, seed=1)
    )

    assert isinstance(result, GenerationResult)
    assert result.prompt == "The purpose of"
    assert isinstance(result.completion, str)
    assert result.text.startswith("The purpose of")
    assert result.prompt_token_count > 0
    assert result.generated_token_count <= 5
    assert len(result.token_ids) == result.generated_token_count


def test_bos_is_prepended_by_default(tokenizer):
    model = Transformer(small_config(vocab_size=tokenizer.vocab_size))

    with_bos = generate(
        model, tokenizer, "hello", SamplingConfig(max_new_tokens=1, add_bos=True)
    )
    without_bos = generate(
        model, tokenizer, "hello", SamplingConfig(max_new_tokens=1, add_bos=False)
    )

    assert with_bos.prompt_token_count == without_bos.prompt_token_count + 1


def test_generation_is_reproducible_under_a_seed(tokenizer):
    model = Transformer(small_config(vocab_size=tokenizer.vocab_size))
    config = SamplingConfig(max_new_tokens=8, temperature=0.9, top_k=10, seed=42)

    first = generate(model, tokenizer, "In Python", config)
    second = generate(model, tokenizer, "In Python", config)

    assert first.token_ids == second.token_ids
    assert first.completion == second.completion


def test_different_seeds_generally_differ(tokenizer):
    model = Transformer(small_config(vocab_size=tokenizer.vocab_size))
    outputs = {
        tuple(
            generate(
                model,
                tokenizer,
                "In Python",
                SamplingConfig(max_new_tokens=10, temperature=1.0, top_k=None, seed=s),
            ).token_ids
        )
        for s in range(5)
    }
    assert len(outputs) > 1


def test_greedy_generation_is_deterministic_without_a_seed(tokenizer):
    model = Transformer(small_config(vocab_size=tokenizer.vocab_size))
    config = SamplingConfig(max_new_tokens=6, temperature=0.0)

    assert (
        generate(model, tokenizer, "Newton", config).token_ids
        == generate(model, tokenizer, "Newton", config).token_ids
    )


def test_undecodable_ids_are_identified(tokenizer):
    """A tiny corpus cannot fill a 512-id budget, so some ids stay unassigned."""

    undecodable = undecodable_token_ids(tokenizer)

    assert all(0 <= i < tokenizer.vocab_size for i in undecodable)
    for token_id in undecodable[:5]:
        with pytest.raises(ValueError):
            tokenizer.decode([token_id])


def test_generated_output_always_decodes(tokenizer):
    """The failure this guards: one bad id aborts a whole generation."""

    model = Transformer(small_config(vocab_size=tokenizer.vocab_size))

    for seed in range(6):
        result = generate(
            model,
            tokenizer,
            "Newton",
            SamplingConfig(max_new_tokens=25, temperature=1.2, top_k=None, seed=seed),
        )
        assert isinstance(result.completion, str)


def test_clean_text_decodes_without_loss(tokenizer):
    ids = tokenizer.encode("In Python, a dictionary", add_bos=False, add_eos=False)
    text, lossy = decode_generated(tokenizer, ids)

    assert lossy is False
    assert "dictionary" in text


def test_a_truncated_multibyte_character_is_trimmed_not_mangled(tokenizer):
    """Cutting mid-character is normal at max_new_tokens; it must not raise."""

    ids = tokenizer.encode("café ∇ Ω", add_bos=False, add_eos=False)

    for cut in range(1, len(ids) + 1):
        text, lossy = decode_generated(tokenizer, ids[:cut])
        assert isinstance(text, str)
        assert lossy is False, cut


def test_empty_ids_decode_to_empty(tokenizer):
    assert decode_generated(tokenizer, []) == ("", False)


def test_special_tokens_are_skipped(tokenizer):
    eos = tokenizer.special_token_to_id["<|eos|>"]
    ids = tokenizer.encode("hello", add_bos=False, add_eos=False) + [eos]

    text, _ = decode_generated(tokenizer, ids, skip_special_tokens=True)
    assert "<|eos|>" not in text


def test_streaming_reassembles_the_same_text(tokenizer):
    ids = tokenizer.encode(
        "In Python, a dictionary maps keys to values.", add_bos=False, add_eos=False
    )
    decoder = StreamingDecoder(tokenizer)

    streamed = "".join(decoder.push(i) for i in ids) + decoder.flush()
    assert streamed == tokenizer.decode(ids)


def test_streaming_holds_back_partial_characters(tokenizer):
    """A multi-byte character must never be emitted half-decoded."""

    ids = tokenizer.encode("café ∇ Ω μF", add_bos=False, add_eos=False)
    decoder = StreamingDecoder(tokenizer)

    pieces = [decoder.push(i) for i in ids]
    pieces.append(decoder.flush())

    assert "".join(pieces) == tokenizer.decode(ids)
    assert "�" not in "".join(pieces)


def test_streaming_emits_incrementally(tokenizer):
    """Text should appear during the stream, not only at flush."""

    ids = tokenizer.encode(
        "The purpose of a voltage regulator", add_bos=False, add_eos=False
    )
    decoder = StreamingDecoder(tokenizer)

    during = "".join(decoder.push(i) for i in ids)
    assert during


def test_streaming_skips_special_tokens(tokenizer):
    eos = tokenizer.special_token_to_id["<|eos|>"]
    decoder = StreamingDecoder(tokenizer)

    assert decoder.push(eos) == ""


def test_streaming_flush_is_empty_when_nothing_is_pending(tokenizer):
    decoder = StreamingDecoder(tokenizer)
    ids = tokenizer.encode("hello world", add_bos=False, add_eos=False)
    for token_id in ids:
        decoder.push(token_id)
    decoder.flush()

    assert decoder.flush() == ""


def test_streaming_matches_generate(tokenizer):
    """The streamed text must equal what the batch decoder produces."""

    model = Transformer(small_config(vocab_size=tokenizer.vocab_size))
    config = SamplingConfig(max_new_tokens=12, temperature=0.0)

    result = generate(model, tokenizer, "Newton", config)

    decoder = StreamingDecoder(tokenizer)
    streamed = "".join(decoder.push(i) for i in result.token_ids) + decoder.flush()

    assert streamed == result.completion


def test_every_generated_id_is_in_range(tokenizer):
    model = Transformer(small_config(vocab_size=tokenizer.vocab_size))
    result = generate(
        model, tokenizer, "A database", SamplingConfig(max_new_tokens=20, seed=3)
    )

    assert all(0 <= i < tokenizer.vocab_size for i in result.token_ids)


def test_generation_runs_past_the_context_boundary():
    """The real failure this guards: RoPE refuses positions beyond its tables."""

    config = small_config(context_length=8, vocab_size=32)
    model = Transformer(config)
    produced, _ = generate_ids(
        model, [1, 2, 3], SamplingConfig(max_new_tokens=20, temperature=0.0)
    )

    assert len(produced) == 20


def test_a_prompt_containing_special_token_text_is_not_parsed(tokenizer):
    """A user typing <|eos|> must not be able to end their own generation."""

    model = Transformer(small_config(vocab_size=tokenizer.vocab_size))
    result = generate(
        model, tokenizer, "say <|eos|> now", SamplingConfig(max_new_tokens=3, seed=1)
    )

    eos_id = tokenizer.special_token_to_id["<|eos|>"]
    prompt_ids = tokenizer.encode(
        "say <|eos|> now", add_bos=True, add_eos=False, parse_special_tokens=False
    )
    assert eos_id not in prompt_ids[1:]
    assert result.prompt_token_count == len(prompt_ids)


def test_non_string_prompt_is_rejected(tokenizer):
    model = Transformer(small_config(vocab_size=tokenizer.vocab_size))
    with pytest.raises(TypeError):
        generate(model, tokenizer, 123, SamplingConfig())


def test_result_serializes(tokenizer):
    model = Transformer(small_config(vocab_size=tokenizer.vocab_size))
    payload = generate(
        model, tokenizer, "Newton", SamplingConfig(max_new_tokens=3, seed=1)
    ).to_dict()

    assert payload["prompt"] == "Newton"
    assert "completion" in payload and "text" in payload
    assert payload["text"].startswith("Newton")
