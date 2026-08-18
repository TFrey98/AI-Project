# About story-model

`README.md` documents *how* to run this project, phase by phase. This
document explains *what* it is and *why* it exists — for anyone opening the
repo who wants the concept before the commands.

## What this is

story-model is a small language model, trained entirely from scratch (no
pretrained weights borrowed from anywhere), being built up in stages into a
system that can play a specific, authored fictional character in
conversation — consistently, across many turns, while respecting what that
character does and doesn't know.

It is not a general-purpose chatbot. It has no built-in knowledge of the
real world beyond whatever patterns it absorbed from its training text, and
it is not trying to answer questions or be broadly helpful. Its one job is:
*given a character's identity, their relationship with the person talking to
them, the current scene, and what they actually remember or have access to,
predict what that character would plausibly say next.*

## Why build one from scratch

Every part of this system — the tokenizer, the attention mechanism, the
training loop, the character-context format — is implemented and trained
in this repository, rather than fine-tuning an existing large model. That's
a deliberate choice: it means every architectural decision (why RoPE
instead of learned position embeddings, why response-only loss masking,
why conversation-level data splits) is visible, understood, and justified
in the code itself, not inherited as an opaque default. The project is as
much about understanding how a character-playing language model actually
works, end to end, as it is about producing one.

## How it works, conceptually

**A general-purpose foundation.** Before it can play any character, the
model first has to learn the shape of English prose — grammar, dialogue
conventions, narrative rhythm. It's pretrained on public-domain literature
(fairy tales, gothic novels, Arthurian romance, adventure fiction) the same
way any language model is pretrained: predict the next piece of text, over
and over, across millions of tokens. This is generic; the model at this
stage has no notion of "characters" at all, just a broad, malleable
understanding of written English.

**A strict, scoped context format.** Every time the character-playing
model is asked to respond, it's shown a structured bundle of information,
not a free-form prompt: the character's stable identity (personality,
voice, goals, fears, secrets, boundaries), the evolving relationship with
whoever they're talking to (trust, affection, respect, unresolved
obligations), the current scene (location, time, what's physically
present), and — critically — only the memories and world facts that
character is actually allowed to know. A secret one character is hiding
from another literally never enters the model's input when it's that other
character's turn to respond. This scoping is enforced in code, not by
asking the model nicely.

**Specialization without forgetting.** Rather than training a whole new
model for character behavior, the pretrained model is *warm-started*:
its already-learned weights are carried forward and its vocabulary is
extended with a small set of new control tokens (marking where the
character's identity, the scene, the conversation, etc. begin), while
everything it already knows about language stays intact. Fine-tuning then
happens with a *response-only* loss — the model is only ever graded on the
words it generates as the character's reply, never on the surrounding
context it was handed, so it isn't rewarded for parroting its own prompt.

**Data quality before scale.** Before any real fine-tuning run, the
training examples are checked against a fixed vocabulary of ten behavioral
categories a good character response might need to demonstrate: staying in
voice, using memory correctly, holding a boundary, admitting uncertainty,
reacting to conflict, and so on. A dataset has to show meaningful coverage
across these categories — not just raw example count — before it's allowed
to train. Afterward, the trained model is evaluated the same way: loss
broken out per category, not one aggregate number, because a model can
look fine on average while being quietly bad at, say, remembering things.

## What it can't do (yet, or ever)

- It has no factual grounding beyond what's explicitly authored into a
  character's world facts — it cannot look anything up.
- It's small enough to train on a single Apple Silicon laptop (a few
  million parameters, not billions), so its fluency and range are nowhere
  near a production-scale LLM.
- Passing the engineering data-quality gate is not the same as a character
  actually being *believable* — that still requires human judgment, which
  hasn't been built into the pipeline yet.
- Every checkpoint produced so far has been trained on deliberately tiny,
  disposable smoke datasets to prove the mechanics work. None of them
  represent a finished, deployable character.

## Where this is headed

The pipeline so far proves every piece works end to end — tokenization,
pretraining, context scoping, warm-starting, response-only fine-tuning,
and behavioral evaluation — but on toy-scale data built to exercise the
mechanism, not to produce a good character. The next real step is
authoring an actual curated dataset of character conversations, large and
varied enough to pass the Phase 19 coverage gate, followed by the
human-in-the-loop behavioral review this project's own evaluation tooling
still defers to.
