# FarmWise demo — Bare chatbot vs. RAG (grounded)

Open http://localhost:8000/chat and toggle **"Bare chatbot"** ↔ **"RAG · grounded"**.
Ask each question in **both** modes, back-to-back, so the contrast lands.
Answers below were captured live from this build (`llama3.2:3b` + `nomic-embed-text`).

Suggested running order: **1 → 6 → 7 → 3 → 5**, with **2, 4, 8, 9** as spares.
Every source in the knowledge base is about Ghana (see `data/knowledge_sources.txt`).

**Timing:** a bare answer takes about 5 to 15 s. A grounded answer takes about
15 to 35 s on this laptop when the memory is free (close IntelliJ, WhatsApp and
other heavy apps first), and up to a minute if it is swapping. Asking the same
question again answers instantly.

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

## 6. Invented numbers vs field trials (strong)
**"Which fertilizer rates gave the highest maize yield responses in trials across Ghana?"**

- 🔴 **Bare:** "According to our knowledge base… **150 kg/ha N, 120 kg/ha P, 60 kg/ha K**" — invents the numbers *and* claims a knowledge base it does not have.
- 🟢 **RAG:** "**60–130 kg N, 60 kg P₂O₅ and 20–60 kg K₂O per hectare** gave the highest maize yield responses." — cited to the **IFDC FERARI** brief, built on about 1,700 trial data points across Ghana's agro-ecological zones.

> This is the advice a farmer would actually spend money on.

## 7. A number the plain model simply does not have
**"How much does Ghana lose every year to post-harvest losses?"**

- 🔴 **Bare:** "couldn't find specific data on Ghana… globally 10–30%" — no Ghana figure.
- 🟢 **RAG:** "Ghana loses about **US$1.9 billion** worth of produce a year to post-harvest losses." — cited to the **Ghana News Agency** (WFP estimate); cereals account for about US$141 million.

## 8. Practical how-to (spare)
**"What spacing and seed rate should I use when planting maize in Ghana?"**

- 🔴 **Bare:** "**20–25 cm between rows**, 30–40 cm between plants, 10–15 kg/ha" — rows far too close.
- 🟢 **RAG:** rows **60–90 cm** apart (the IITA guide says 75 cm), **8–10 kg of seed per acre (20–25 kg/ha)** — cited to the **IITA Guide for Sustainable Maize Production in Ghana** and the **IPA/CSIR-SARI extension handbook**.

## 9. Government programme fact (spare)
**"In the 2017-18 season, how many farmers did the Planting for Food and Jobs campaign help to increase their maize and rice harvests?"**

- 🔴 **Bare:** "I don't have information on a specific Planting for Food and Jobs campaign."
- 🟢 **RAG:** "**Over 200,000 farmers** in the 2017-18 cropping season." — cited to the **SAIRLA** policy brief and the **World Bank** policy note.

> Ask it with the season in the question. Asked as "in its first season", retrieval
> missed it and the assistant said it did not know.

---

## Avoid these on stage

Tested on 9 November and not reliable enough to demo:

- **Hermetic (PICS) bag storage**: the grounded answer mixed two sources and wrongly
  said the bags should be sprayed with insecticide.
- **When to top-dress maize with urea**: the sources give different timings (about
  35 days and 6 weeks after planting), so the answer depends on which passage wins.
- **Fall armyworm control**: the source is a field study, and the answer did not
  give clear control steps.
- **Cocoa fermentation and drying**: the answer was vague (it refers to a table).
- **Questions in Twi, Fante, Ewe or Ga**: the 3B model mixes in other languages.

---

Tip: answers from a 3B local model vary slightly run-to-run; the *contrast* is stable,
the exact wording is not. If a bare answer comes out unusually sensible, re-ask — or
fall back to question 1, which is the most reliable.
