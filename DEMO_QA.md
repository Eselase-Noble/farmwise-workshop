# FarmWise demo — Bare chatbot vs. RAG (grounded)

Open http://localhost:8000/chat and toggle **"Bare chatbot"** ↔ **"RAG · grounded"**.
Ask each question in **both** modes, back-to-back, so the contrast lands.
Answers below were captured live from this build (`llama3.2:3b` + `nomic-embed-text`).

Suggested running order: **1 → 2 → 3 → 5**, with **4** as a spare.

---

## 1. Hard number — the money shot
**"Which single region produces the most maize in Ghana, and roughly how many tonnes per year?"**

- 🔴 **Bare:** "Ashanti Region… around **1.2 million tonnes** per year." — right region, **invented number (~2.5× too high)**, no source.
- 🟢 **RAG:** "Ashanti Region… **476,678 metric tonnes** (three-year average)." — **cited to MoFA Facts & Figures.**

> Same confident tone — but one number is made up and one is from the national statistics office.

## 2. Wrong fact stated confidently
**"What does agriculture contribute to employment in Ghana, and which crops dominate?"**

- 🔴 **Bare:** "about **20%** of the workforce, ~1.5 million people" — **wrong.**
- 🟢 **RAG:** "**38.3%** of total employment, second largest employer after Services" — cited.

## 3. The one we taught it live (great story)
**"Which regions of Ghana are the main producers of cocoa and maize?"**

- 🔴 **Bare:** names wrong districts ("Cape Coast, Elmina" — not cocoa areas).
- 🟢 **RAG:** "Cocoa: Western & Western North (largest), plus Ashanti, Ahafo, Central, Eastern, Volta. Maize: Ashanti and Eastern, then Bono, Central, Bono East." — cited.

> This data lived only in a PDF chart, so the assistant used to decline. We added one
> sourced note (`data/guides/regional_producers.md`) to the knowledge base — now it
> answers. That's the `/knowledge` page: the assistant only knows what it has read.

## 4. Precision + citation (softer contrast — spare)
**"How important is cocoa to Ghana's economy and where does it rank as a producer?"**

- 🔴 **Bare:** "2nd largest, ~20–30% of global production" — roughly right, no source.
- 🟢 **RAG:** "**20–25% of export earnings, 7% of GDP**, world's 2nd-largest producer" — cited to **World Bank**.

## 5. Honesty / no bluffing (the trust point — close on this)
**"What is the average price of tomatoes at the market in Kumasi today?"**

- 🔴 **Bare:** rambles a list of places to go look.
- 🟢 **RAG:** "**I don't know based on the farming guides provided.**"

> A grounded assistant admits what it doesn't know instead of guessing — exactly what a farmer needs.

---

Tip: answers from a 3B local model vary slightly run-to-run; the *contrast* is stable,
the exact wording is not. If a bare answer comes out unusually sensible, re-ask — or
fall back to question 1, which is the most reliable.
