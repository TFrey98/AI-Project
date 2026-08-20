# Phase 22: foundation corpus v2

## Decision

The 2.86M-parameter foundation was trained on about 4.78M BPE tokens.  That is
only about 1.67 corpus tokens per parameter, and its single-digit number of
training authors is too narrow for reliable prose semantics.  Phase 22 expands
and diversifies the language foundation before Vera-specific authoring resumes.

The run is deliberately a data-only continuation:

- same 2,863,616-parameter Transformer;
- same learned 512-token byte-BPE vocabulary;
- same 256-token context and 4,096 sampled tokens per update;
- no character control tokens;
- lower `1e-4` peak learning rate for continued pretraining.

Changing model size, tokenizer, and corpus together would make it impossible to
tell which change helped.  Longer context and a larger model remain later,
separate experiments.

## Corpus target

The production gates enforce at least:

| Measure | Gate | Preferred target |
|---|---:|---:|
| Normalized training text | 100 MB | 110-135 MB |
| Training BPE tokens | 50 million | 55-65 million |
| Works | 50 | 75-150 |
| Authors | 25 | 35-60 |
| Validation authors | 5 | 6-10 |
| Broad categories | 5 | 6 or more |
| Largest training author | at most 10% of bytes | 5-8% |
| Corpus tokens / model parameter | at least 15 | 18-22 |

The 30,000-update full config samples 122,880,000 training tokens, roughly two
passes over a 55-65M-token corpus.  Sampling is random rather than epoch-based,
so this is an exposure estimate rather than an exact pass count.

## Selection mix

Favor works with interaction, intention, consequence, and sustained scenes.
The following byte shares are curation targets, not mutually exclusive labels:

| Material | Approximate share | Purpose |
|---|---:|---|
| Dialogue-rich novels and novellas | 30-40% | Turn-taking, subtext, relationships |
| Plays and dramatic works | 15-20% | Dense speech, conflict, distinct voices |
| Mystery, gothic, and suspense | 15-20% | Evidence, uncertainty, secrets, tension |
| Adventure and historical fiction | 15-20% | Goals, plans, action grounded in scenes |
| Letters, diaries, and first-person narratives | 10-15% | Direct address, reflection, stable voice |
| Speculative and folkloric fiction | 10-15% | Non-modern worlds and invented situations |

Avoid letting Shakespeare, Arthurian prose, or any single highly archaic style
dominate.  A useful period balance is roughly 70% nineteenth/early-twentieth
century prose, 20% drama, and no more than 10% older prose.  English
translations are acceptable but should name both the original author and the
translator in catalog notes when known.

Candidate authors include Jane Austen, Charlotte/Emily/Anne Bronte, George
Eliot, Elizabeth Gaskell, Charles Dickens, Thomas Hardy, Anthony Trollope,
Wilkie Collins, Arthur Conan Doyle, Robert Louis Stevenson, H. G. Wells, Bram
Stoker, Sheridan Le Fanu, Mary Shelley, Mark Twain, Louisa May Alcott, Frances
Hodgson Burnett, L. M. Montgomery, Jack London, Joseph Conrad, G. K.
Chesterton, Baroness Orczy, Edith Wharton, Henry James, Oscar Wilde, George
Bernard Shaw, Richard Brinsley Sheridan, and several additional authors so the
author cap is genuinely met.

## Source and rights workflow

Use public-domain or otherwise clearly authorized text only.  Project
Gutenberg's normal website is intended for human browsing; its
[robot-access policy](https://www.gutenberg.org/policy/robot_access.html)
documents the supported harvest endpoint for automation.  Its
[copyright guidance](https://www.gutenberg.org/help/copyright.html) explains
that rights can differ outside the United States.  Record a rights note and
source URL for every work and verify the deployment jurisdiction rather than
assuming that an old publication date is sufficient.

The official bulk command is:

```bash
wget -w 2 -m -H \
  "https://www.gutenberg.org/robot/harvest?filetypes[]=txt&langs[]=en"
```

That mirror can be very large.  For a 125-150 MB curated corpus, it is usually
more practical to use the catalog to choose works first and download the
selected UTF-8 text files manually.  Do not write a scraper against ordinary
ebook pages.

Place files under author directories such as:

```text
data/raw_v2/austen/pride_and_prejudice.txt
data/raw_v2/stevenson/treasure_island.txt
```

Copy the schema example and fill one record per file:

```bash
cp examples/foundation_corpus_catalog.json \
  corpus_catalogs/foundation_v2.json
```

`author_id` is the split identity.  Every work by the same human author must
use exactly the same lowercase ID.  `source_id` is the source's stable ebook
identifier.  Categories are broad lowercase labels such as `dialogue`,
`drama`, `relationship`, `mystery`, `gothic`, `adventure`, `historical`,
`first_person`, or `speculative`.

## Build and gate

```bash
python scripts/build_foundation_corpus.py
```

The command rejects uncataloged/missing files, duplicate content, duplicate
source IDs, author leakage, insufficient scale/diversity, and an author that
exceeds 10% of training bytes.  It writes:

```text
data/processed_v2/train.txt
data/processed_v2/val.txt
data/processed_v2/manifest.json
```

Next count tokens with the original pre-character foundation tokenizer:

```bash
python scripts/audit_foundation_corpus.py \
  --checkpoint checkpoints/transformer_bpe_medium/best.pt
```

The attached `transformer_character_bridge/best.pt` is not valid here.  It has
nine character-control tokens and behavior-specialized weights; the audit and
training config both reject it.  If `transformer_bpe_medium/best.pt` no longer
exists, retrain Phase 13 before Phase 22 rather than trying to reverse a bridge
checkpoint.

## Train

Run the 200-update wiring smoke test:

```bash
python -m story_model.train \
  --config configs/transformer_foundation_v2_smoke.yaml \
  --warm-start checkpoints/transformer_bpe_medium/best.pt
```

Then run the full continuation:

```bash
python -m story_model.train \
  --config configs/transformer_foundation_v2.yaml \
  --warm-start checkpoints/transformer_bpe_medium/best.pt
```

Use `checkpoints/transformer_foundation_v2/best.pt`, not `final.pt`, as the
candidate new foundation.  Compare old and new checkpoints on the identical v2
validation split before retraining the generic dialogue bridge:

```bash
python -m story_model.evaluate \
  --checkpoint checkpoints/transformer_bpe_medium/best.pt \
  --data-config configs/transformer_foundation_v2.yaml \
  --split val

python -m story_model.evaluate \
  --checkpoint checkpoints/transformer_foundation_v2/best.pt \
  --data-config configs/transformer_foundation_v2.yaml \
  --split val
```

The new foundation should lower held-out bits/byte, not merely training loss.
After it passes fixed-generation review, rebuild Phase 21B from the new
foundation checkpoint.  Vera-specific fine-tuning and durable conversation
memory remain downstream layers; more literature alone will not teach Vera's
canon or memory policy.
