# data/guides/ — Farming guide documents (RAG knowledge base)

The RAG assistant answers farmer questions **only** from the documents in this
folder. Supported formats: `.pdf`, `.docx`, `.xlsx`, `.xls`, `.csv`, `.html`,
`.htm`, `.txt`, `.md`.

## Real guides (presenter supplies)

Drop real, public extension materials here before the session — e.g. MoFA
(Ghana Ministry of Food and Agriculture) or FAO crop/extension guides. Then
rebuild the index:

```
python scripts/build_index.py
```

## ⚠️ Sample files in this folder

The files named `sample_*.md` are **SAMPLE – FOR TESTING ONLY, NOT AGRONOMIC
ADVICE**. They exist so the RAG pipeline can be built and tested before real
guides are added. Each one carries that disclaimer in its header.

**Delete or replace the sample files with real guides before presenting.** The
Session 2 web app already leaves them out of the index unless
`FARMWISE_INCLUDE_SAMPLES=1` is set.
They contain simplified, illustrative content and must never be presented as
real farming advice.
