# Phase 25: coherent neutral conversation backbone

## Purpose

Foundation-v3 remains the project's from-scratch language-model checkpoint,
but its qualitative trial showed that it cannot yet provide dependable scene
meaning or question answering.  Phase 25 separates two acceptance gates:

1. **Neutral conversational competence:** logical, relevant answers grounded in
   supplied context.
2. **Vera fidelity:** Vera's voice, motivations, deception, relationships, and
   behavioral consistency.

This phase implements only the first gate.  It does not train Vera and does not
replace foundation-v3.  Instead, character state can now be sent through an
interchangeable `LanguageBackbone` interface:

- `ScriptedBackbone` returns deterministic responses for unit tests;
- the existing checkpoint generator continues to test from-scratch model code;
- `LocalOpenAIBackbone` calls a coherent instruction model running on the same
  computer through an OpenAI-compatible `/chat/completions` endpoint.

The local HTTP adapter accepts only `localhost` or loopback IP addresses.  A
mistyped remote or LAN URL is rejected before any character context, memory, or
conversation can be transmitted.

## Local model server

Start a local instruction model using a server that exposes an OpenAI-compatible
chat-completions endpoint.  Common default endpoint shapes include:

| Local runtime | Endpoint |
|---|---|
| Ollama compatibility API | `http://127.0.0.1:11434/v1/chat/completions` |
| llama.cpp server | `http://127.0.0.1:8080/v1/chat/completions` |
| LM Studio local server | `http://127.0.0.1:1234/v1/chat/completions` |

Model installation and server startup are runtime-specific.  Keep the server
bound to loopback and substitute its exact local model identifier for
`<local-model-name>` below.

## Test gate

Run the project tests first:

```bash
.venv/bin/python -m pytest -q
```

Then run the seven-case neutral conversation gate:

```bash
.venv/bin/python scripts/evaluate_conversation_gate.py \
  --model <local-model-name> \
  --endpoint http://127.0.0.1:11434/v1/chat/completions \
  --output runs/phase25-neutral-gate.jsonl
```

The automatic checks cover:

- a route constrained by the current scene;
- direct recovery of a supplied fact;
- refusal to invent missing information;
- concrete memory across intervening turns;
- correction of a false premise;
- pronoun/reference tracking;
- cause and effect.

Automatic phrase checks are a fast wiring gate, not a complete quality score.
Every case also prints a manual criterion.  Accept a model only when it passes
all seven automatic checks and each response is concise, logical, and relevant
under manual review.

## Interactive character-context test

Use the existing provisional context with the neutral backbone:

```bash
.venv/bin/python scripts/chat_local_instruct.py \
  --model <local-model-name> \
  --endpoint http://127.0.0.1:11434/v1/chat/completions \
  --context examples/character_context.json
```

The adapter converts the existing `CharacterContext` into a normal system
message plus user/assistant turns.  Only memories owned by or shared with the
active character and only public/known world facts are included.  Unknown or
other-character-private information remains excluded exactly as it is in the
from-scratch control-token path.

At this stage the character profile is merely context.  Passing the neutral
gate means the backbone can answer questions and follow supplied state; it does
not mean that the model has learned Vera.  Vera-specific training and
evaluation begin only after her standardized dialogue set exists.
