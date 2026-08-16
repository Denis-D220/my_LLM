"""Builder-level tests for evaluation decontamination in production.

The unit tests in ``test_decontamination.py`` cover the detector.  These cover
the thing that actually protects the benchmark: that the corpus builder refuses
to produce a training corpus which contains evaluation material, refuses to
produce one silently unprotected, and refuses to mix documents processed under
different rules.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from llm.data.decontamination import DecontaminationConfig
from llm.data.pretraining_corpus import (
    MANIFEST_NAME,
    PROGRESS_NAME,
    REASON_EVAL_CONTAMINATION_CONTAINMENT,
    REASON_EVAL_CONTAMINATION_EXACT,
    build_pretraining_corpus,
    file_sha256,
)


EVAL_PASSAGE = (
    "A switching regulator transfers energy in discrete packets rather than "
    "dissipating the difference between input and output as heat. During the "
    "on interval the inductor current rises and stores magnetic energy, and "
    "during the off interval that stored energy is delivered to the output "
    "through the freewheeling path. The ratio of on time to the full switching "
    "period determines the conversion ratio, and feedback adjusts that ratio "
    "so the output stays at the programmed value as load and input vary. "
    "Efficiency is highest near the middle of the load range because "
    "conduction losses dominate at high current while switching and quiescent "
    "losses dominate at light load."
)

UNRELATED = (
    "Warehouse pallet racking is specified by beam capacity, upright depth, "
    "and the clear entry height required by the handling equipment in use. "
    "Aisle width follows from the turning radius of the trucks rather than "
    "from the racking itself, and a narrow aisle installation trades floor "
    "area against the cost of guided or articulated trucks. Seismic zones add "
    "base plate and anchor requirements that frequently govern the design of "
    "the upright frames, and inspection regimes are set by the operator. "
    "Load notices must state the configuration they apply to, because a "
    "change in beam level pitch changes the permissible unit load."
)

SAME_SUBJECT_DIFFERENT_PROSE = (
    "Buck converters step voltage down by chopping the supply with a "
    "transistor and smoothing the result with an output filter. Designers "
    "size that filter for acceptable ripple, pick a switching frequency that "
    "balances magnetics against gate losses, and confirm the control loop is "
    "stable across the operating envelope. Thermal behaviour is usually "
    "verified last and on real hardware, because copper area dominates the "
    "outcome and is awkward to model with confidence beforehand."
)


# Filler must satisfy two constraints that pull in opposite directions. It has
# to contribute many distinct shingles, so the embedded evaluation passage is a
# small fraction of the page and symmetric similarity stays low. It also has to
# survive the quality stage, which runs *before* decontamination: repetitive
# filler is a keyword wall and gets rejected earlier, so the test would pass for
# entirely the wrong reason.
_HEAD = (
    "bar", "cal", "dor", "fen", "gil", "hap", "jor", "kel",
    "mor", "nes", "pol", "ral", "sun", "tor", "vel", "wyn",
)
_TAIL = (
    "ford", "wick", "dale", "mere", "stow", "holt", "bury", "gate",
    "combe", "thorpe", "field", "ridge", "brook", "hall", "worth", "leigh",
)
_ACTIONS = (
    "logged", "revised", "approved", "deferred", "escalated",
    "annotated", "reinstated", "withdrew", "circulated", "questioned",
)
_OBJECTS = (
    "drainage schedule", "culvert survey", "embankment profile",
    "ballast specification", "signalling plan", "fencing register",
    "vegetation report", "crossing study", "deck log", "earthworks appraisal",
)


def _coined(n: int) -> str:
    """Distinct pronounceable token, so each line adds real vocabulary."""

    return _HEAD[n % 16] + _TAIL[(n // 16) % 16]


def page_filler(start: int = 0, stop: int = 50) -> str:
    """Filler that is long, distinct, and passes the quality stage.

    Combining a fixed word pool is not enough: repeating the same few hundred
    words drives the type-token ratio below the keyword-wall threshold, and the
    page is rejected by quality *before* decontamination ever runs.  Each line
    therefore contributes genuinely new tokens.
    """

    return "\n".join(
        f"{_coined(n).capitalize()} depot {_ACTIONS[n % 10]} the "
        f"{_OBJECTS[(n // 3) % 10]} for {_coined(n + 137)} on entry {n:04d}."
        for n in range(start, stop)
    )


def write_evaluation(path: Path, items: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in items) + "\n",
        encoding="utf-8",
    )
    return path


def default_evaluation(tmp_path: Path) -> Path:
    return write_evaluation(
        tmp_path / "eval" / "pretraining_eval.jsonl",
        [
            {
                "id": "eval-eng-0001",
                "category": "engineering",
                "text": EVAL_PASSAGE,
            }
        ],
    )


def write_source(path: Path, texts: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for index, text in enumerate(texts):
            handle.write(
                json.dumps(
                    {
                        "text": text,
                        "url": f"https://example.com/{index}",
                        "date": "2026-08-14T00:00:00Z",
                        "lang": "eng",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


def read_documents(output: Path) -> list[dict]:
    documents: list[dict] = []
    for shard in sorted(output.glob("corpus-*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    documents.append(json.loads(line))
    return documents


def build(tmp_path: Path, texts: list[str], **kwargs):
    source = tmp_path / "source"
    write_source(source / "cc_text_00000.jsonl.gz", texts)
    kwargs.setdefault("evaluation_path", default_evaluation(tmp_path))
    return build_pretraining_corpus(
        source=source,
        output=kwargs.pop("output", tmp_path / "out"),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# The eight required builder-level behaviours
# ---------------------------------------------------------------------------


def test_filler_survives_the_quality_stage(tmp_path: Path) -> None:
    """Guards the fixture itself.

    Quality filtering runs before decontamination. If the filler were rejected
    as a keyword wall, the embedded-passage test would pass for entirely the
    wrong reason and would prove nothing about contamination.
    """

    from llm.data.quality import assess_document, compute_metrics

    page = page_filler(0, 60) + "\n\n" + page_filler(60, 120)
    metrics = compute_metrics(page)

    assert metrics.type_token_ratio > 0.10
    assert assess_document(page).accepted, assess_document(page).reason


def test_unrelated_document_is_accepted(tmp_path: Path) -> None:
    report = build(tmp_path, [UNRELATED])
    assert report["stats"]["documents_accepted"] == 1


def test_exact_evaluation_document_is_rejected(tmp_path: Path) -> None:
    report = build(tmp_path, [EVAL_PASSAGE, UNRELATED])

    assert report["stats"]["documents_accepted"] == 1
    assert (
        report["stats"]["rejected_by_reason"][REASON_EVAL_CONTAMINATION_EXACT]
        == 1
    )
    texts = [doc["text"] for doc in read_documents(tmp_path / "out")]
    assert all(EVAL_PASSAGE not in text for text in texts)


def test_evaluation_passage_embedded_in_long_page_is_rejected(
    tmp_path: Path,
) -> None:
    # Two disjoint ranges: reusing one range would create duplicate lines and
    # the document would be rejected by the duplicate-line rule instead.
    page = (
        page_filler(0, 60)
        + "\n\n"
        + EVAL_PASSAGE
        + "\n\n"
        + page_filler(60, 120)
    )
    report = build(tmp_path, [page, UNRELATED])

    assert report["stats"]["documents_accepted"] == 1
    assert (
        report["stats"]["rejected_by_reason"][
            REASON_EVAL_CONTAMINATION_CONTAINMENT
        ]
        == 1
    ), f"unexpected rejections: {report['stats']['rejected_by_reason']}"


def test_same_subject_different_prose_is_accepted(tmp_path: Path) -> None:
    report = build(tmp_path, [SAME_SUBJECT_DIFFERENT_PROSE])
    assert report["stats"]["documents_accepted"] == 1
    assert not any(
        reason.startswith("evaluation_contamination")
        for reason in report["stats"]["rejected_by_reason"]
    )


def test_decontamination_disabled_explicitly_accepts_evaluation_text(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    write_source(source / "cc_text_00000.jsonl.gz", [EVAL_PASSAGE])

    report = build_pretraining_corpus(
        source=source,
        output=tmp_path / "out",
        check_decontamination=False,
    )

    assert report["stats"]["documents_accepted"] == 1


def test_missing_evaluation_artifact_fails_the_build(tmp_path: Path) -> None:
    """Fail closed: omission must not silently produce an unprotected corpus."""

    source = tmp_path / "source"
    write_source(source / "cc_text_00000.jsonl.gz", [UNRELATED])

    with pytest.raises(ValueError, match="evaluation"):
        build_pretraining_corpus(source=source, output=tmp_path / "out")


def test_nonexistent_evaluation_file_fails_the_build(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_source(source / "cc_text_00000.jsonl.gz", [UNRELATED])

    with pytest.raises(FileNotFoundError):
        build_pretraining_corpus(
            source=source,
            output=tmp_path / "out",
            evaluation_path=tmp_path / "absent.jsonl",
        )


def test_manifest_binds_the_corpus_to_the_evaluation_artifact(
    tmp_path: Path,
) -> None:
    evaluation = default_evaluation(tmp_path)
    build(tmp_path, [UNRELATED], evaluation_path=evaluation)

    manifest = json.loads(
        (tmp_path / "out" / MANIFEST_NAME).read_text(encoding="utf-8")
    )
    block = manifest["filters"]["evaluation_decontamination"]

    assert block["enabled"] is True
    assert block["evaluation_sha256"] == file_sha256(evaluation)
    assert len(block["evaluation_sha256"]) == 64
    assert block["evaluation_items"] == 1
    assert block["min_containment"] == 0.80
    assert block["shingle_size"] == 8
    assert block["min_evaluation_tokens"] == 50


def test_resume_preserves_evaluation_configuration(tmp_path: Path) -> None:
    source = tmp_path / "source"
    evaluation = default_evaluation(tmp_path)
    output = tmp_path / "out"

    write_source(source / "cc_text_00000.jsonl.gz", [UNRELATED])
    build_pretraining_corpus(
        source=source,
        output=output,
        evaluation_path=evaluation,
    )

    write_source(
        source / "cc_text_00001.jsonl.gz",
        [SAME_SUBJECT_DIFFERENT_PROSE],
    )
    build_pretraining_corpus(
        source=source,
        output=output,
        evaluation_path=evaluation,
        resume=True,
    )

    progress = json.loads(
        (output / PROGRESS_NAME).read_text(encoding="utf-8")
    )
    manifest = json.loads((output / MANIFEST_NAME).read_text(encoding="utf-8"))

    assert progress["decontamination"]["evaluation_sha256"] == file_sha256(
        evaluation
    )
    assert manifest["filters"]["evaluation_decontamination"][
        "evaluation_sha256"
    ] == file_sha256(evaluation)
    assert len(read_documents(output)) == 2


# ---------------------------------------------------------------------------
# Configuration drift
# ---------------------------------------------------------------------------


def start_build(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source"
    evaluation = default_evaluation(tmp_path)
    output = tmp_path / "out"

    write_source(source / "cc_text_00000.jsonl.gz", [UNRELATED])
    build_pretraining_corpus(
        source=source,
        output=output,
        evaluation_path=evaluation,
    )
    write_source(source / "cc_text_00001.jsonl.gz", [UNRELATED + " Second."])
    return source, evaluation, output


def test_resume_rejects_changed_containment_threshold(tmp_path: Path) -> None:
    source, evaluation, output = start_build(tmp_path)

    with pytest.raises(ValueError, match="decontamination policy changed"):
        build_pretraining_corpus(
            source=source,
            output=output,
            evaluation_path=evaluation,
            decontamination_config=DecontaminationConfig(min_containment=0.90),
            resume=True,
        )


def test_resume_rejects_changed_evaluation_file(tmp_path: Path) -> None:
    """A single edited byte in the evaluation set must stop a resume."""

    source, evaluation, output = start_build(tmp_path)

    write_evaluation(
        evaluation,
        [
            {"id": "eval-eng-0001", "category": "engineering", "text": EVAL_PASSAGE},
            {
                "id": "eval-eng-0002",
                "category": "engineering",
                "text": SAME_SUBJECT_DIFFERENT_PROSE,
            },
        ],
    )

    with pytest.raises(ValueError, match="decontamination policy changed"):
        build_pretraining_corpus(
            source=source,
            output=output,
            evaluation_path=evaluation,
            resume=True,
        )


def test_resume_rejects_disabling_decontamination(tmp_path: Path) -> None:
    source, _evaluation, output = start_build(tmp_path)

    with pytest.raises(ValueError, match="decontamination policy changed"):
        build_pretraining_corpus(
            source=source,
            output=output,
            check_decontamination=False,
            resume=True,
        )


def test_resume_rejects_changed_shingle_size(tmp_path: Path) -> None:
    source, evaluation, output = start_build(tmp_path)

    with pytest.raises(ValueError, match="decontamination policy changed"):
        build_pretraining_corpus(
            source=source,
            output=output,
            evaluation_path=evaluation,
            decontamination_config=DecontaminationConfig(shingle_size=10),
            resume=True,
        )


def test_relocating_the_evaluation_file_does_not_block_resume(
    tmp_path: Path,
) -> None:
    """Moving the artifact changes no decision; editing it does."""

    source, evaluation, output = start_build(tmp_path)

    moved = tmp_path / "elsewhere" / "pretraining_eval.jsonl"
    moved.parent.mkdir(parents=True, exist_ok=True)
    moved.write_bytes(evaluation.read_bytes())

    report = build_pretraining_corpus(
        source=source,
        output=output,
        evaluation_path=moved,
        resume=True,
    )
    assert report["stats"]["documents_accepted"] == 1
