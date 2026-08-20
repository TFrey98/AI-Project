# Foundation corpus catalogs

Keep the reviewed source inventory here even though the downloaded text and
processed corpus under `data/` remain outside Git.  A tracked catalog makes a
lost machine rebuildable: it records the exact source ID and URL, normalized
relative path, author grouping, rights note, and content categories for every
work.

Copy `examples/foundation_corpus_catalog.json` to
`corpus_catalogs/foundation_v2.json`, then replace its two illustrative records
with the complete reviewed inventory.  Do not leave the example records in the
production catalog unless those exact files are present in `data/raw_v2/`.

The builder requires a one-to-one match between catalog records and `.txt`
files.  Commit the completed catalog, not the downloaded books, processed text,
or checkpoints.
