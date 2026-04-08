# Product Deduplication Pipeline

A hybrid rule-based + semantic + LLM-assisted system for deduplicating multilingual eCommerce product listings and surfacing the correct lowest valid price per product group.

## The Problem

Zap aggregates listings from multiple stores. The same physical product appears under different titles across languages, token orders, and formatting conventions — making it impossible to group listings or show a reliable lowest price without deduplication.

```
listing_id  title                                price
1           Samsung Galaxy S23 128GB Black       ₪3,199
2           סמסונג גלקסי S23 שחור 128 גיגה        ₪3,099
3           Samsung S23 Galaxy 128GB Black       ₪3,150
55          Samsung Galaxy S23 128GB Black       ₪100   ← suspicious outlier
```

Three challenges the pipeline addresses:

- **Multilingual titles** — same product listed in Hebrew by one store, English by another
- **Variant ambiguity** — products that look similar but are distinct SKUs (`S23` vs `S23 Ultra`, `iPhone 15` vs `iPhone 15 Pro`)
- **Dirty pricing** — outlier prices that would corrupt the displayed minimum if not filtered

---

## Pipeline Architecture

```
Raw CSV input
    │
    ▼
Data Quality Flagging       missing titles, invalid prices, suspicious outliers
    │
    ▼
Attribute Extraction        structured fields from noisy multilingual titles
    │
    ▼
Blocking                    candidate pair generation 
    │
    ▼
Multi-Signal Scoring        fuzzy + attribute agreement + embedding similarity
    │
    ▼
Decision Engine
    ├── Hard conflict   →   Reject immediately
    ├── Strong match    →   Merge immediately
    └── Borderline      →   LLM verification
                                │
                                ▼
                           Post-LLM variant guardrail
    │
    ▼
Union-Find Grouping         transitive closure of merge decisions
    │
    ▼
Canonical title + min price  suspicious price filtering applied here
    │
    ▼
Output CSVs + Evaluation
```

---

## File Structure

```
├── main.py                   entry point — orchestrates the full pipeline
├── attribute_extraction.py   regex + alias-based field extraction from titles
├── matching.py               blocking, scoring, and decision engine
├── llm_layer.py              LLM prompt construction and response parsing
├── postprocessing.py         Union-Find grouping and canonical selection
├── evaluation.py             metrics against gold labels
├── analysis_layer.py         LLM usage and error analysis outputs
├── config.py                 tunable thresholds for matching behavior
├── .env                      API keys (gitignored)
├── .gitignore                excludes .env and outputs/*.csv
├── data/
│   ├── products.csv          input listings
│   └── gold_labels.csv       labeled pairs for evaluation
└── outputs/                  gitignored — generated at runtime
    ├── products_with_attributes.csv
    ├── match_decisions.csv
    ├── grouped_products.csv
    ├── llm_error_analysis.csv
    └── llm_usage_summary.csv
```

---

## Setup

```bash
pip install pandas rapidfuzz sentence-transformers scikit-learn openai python-dotenv
```

Create a `.env` file in the project root:

```
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=your_model_here
```

Run:

```bash
python main.py
```

Input expects a CSV at `data/products.csv` with columns:
`listing_id`, `title`, `price`, `source`, `currency`

Optional: `data/gold_labels.csv` with columns `listing_id_1`, `listing_id_2`, `label` (1 = match, 0 = non-match) for evaluation.

---

## Components

### 1. Data Quality Flagging — `main.py`

Every row is flagged before matching:

| Flag | Condition |
|------|-----------|
| `dq_missing_title` | blank or null title |
| `dq_missing_price` | null price |
| `dq_nonpositive_price` | price ≤ 0 |
| `dq_suspicious_low_price` | price < ₪200 |

Rows with missing title or invalid price are excluded from matching entirely. They are retained in `products_with_attributes.csv` with flags for downstream review.

Listing 55 (₪100 Samsung S23) is grouped with the correct product cluster, flagged as suspicious, and excluded from the displayed minimum price and canonical title selection.

---

### 2. Attribute Extraction — `attribute_extraction.py`

Extracts structured fields from raw title strings using a Hebrew↔English alias dictionary and regex patterns, with a normalization step that handles token reordering, Hebrew aliases, storage unit standardization (TB → GB), and generation notation (`2nd gen` → `gen2`).

Fields extracted:

| Field | Examples |
|-------|---------|
| `brand` | samsung, apple, xiaomi, lenovo, sony, nintendo |
| `model_family` | galaxy, iphone, airpods, ideapad, watch, switch |
| `model_number` | s23, 15, gen2, xm5 |
| `tier_variant` | pro, ultra, plus, fe |
| `display_variant` | oled, standard |
| `connectivity` | gps, cellular, gps_cellular |
| `condition` | new, refurbished, used |
| `storage` | 128gb, 256gb, 512gb |
| `ram` | 8gb, 16gb |
| `screen_size` | 15, 41mm, 45mm |
| `color` | black, white, mint, midnight |

Each field carries a confidence flag (`high` / `missing`) used in the scoring layer.

---

### 3. Blocking — `matching.py`

Each listing is assigned composite block keys via `build_block_keys()`. Only pairs sharing at least one key are evaluated. Multiple blocking strategies combine brand, model family, model number, storage, RAM, screen size, condition, tier variant, display variant, and connectivity. A token-based fallback handles listings where attribute extraction produces no keys.

**55 listings → 197 candidate pairs** (vs 1,485 brute-force)

---

### 4. Multi-Signal Scoring — `matching.py`

```python
final_score = 0.50 * fuzzy_score
            + 0.35 * attribute_agreement
            + 0.15 * embedding_score
```

| Signal | Weight | Rationale |
|--------|--------|-----------|
| Fuzzy similarity | 50% | Token overlap is the strongest single signal for mixed-language listings, even when attribute extraction is incomplete. Computed as average of token sort ratio, token set ratio, and partial ratio via RapidFuzz. |
| Attribute agreement | 35% | Fraction of fields where both listings have extracted values that match exactly. High precision when coverage is good; lower weight because sparse extraction reduces coverage. |
| Embedding similarity | 15% | Cosine similarity via `paraphrase-multilingual-MiniLM-L12-v2`. Useful for cross-language pairs; kept low because semantically similar model names (S23 vs S23 Ultra) would otherwise inflate scores. |

The current weights were chosen based on signal reliability in this dataset and reviewed against the gold label set. Production deployment would require systematic tuning on a larger labeled corpus.

---

### 5. Decision Engine — `matching.py`

**Tier 1 — Hard conflict rejection**
If two listings have explicitly different values for any of: brand, model family, model number, storage, tier variant, display variant, connectivity, RAM, or condition — rejected immediately without computing scores.

**Tier 2 — Strong identity merge**
If normalized titles are identical, or all core identity fields (brand, model family, model number, storage, tier variant, condition, color) match exactly — merged without LLM involvement.

**Tier 3 — LLM verification**
Pairs in the borderline decision zone, or pairs where a risk flag is detected (for example, one side has a special tier variant and the other does not), are routed to the LLM for verification.

---

### 6. LLM Verification — `llm_layer.py`

The LLM receives structured input rather than raw title strings — extracted attributes, all three similarity scores, routing reason, and the pre-LLM rule decision:

```json
{
  "product_a": { "brand": "samsung", "model_number": "s23", "tier_variant": null, "storage": "128gb" },
  "product_b": { "brand": "samsung", "model_number": "s23", "tier_variant": "fe",  "storage": "128gb" },
  "evidence": {"fuzzy_score": 0.85,"attribute_score": 0.80,"embedding_score": 0.83,"final_score": 0.83,"route": "llm_verify"
}
}
```

Output normalized to: `{ "match": "yes/no", "confidence": 0.9/0.6/0.3, "reason": "..." }`

The system prompt instructs the model to default to no under uncertainty. LLM verification was applied to **36 of 197 pairs (18.27%)**.

---

### 7. Variant Guardrails — `main.py`

After an LLM `yes` decision, `is_dangerous_variant_mismatch()` runs before any merge is committed. It rejects if:

- both sides have different explicit tier variants (`pro` vs `ultra`)
- one side carries a special variant (`pro`, `ultra`, `plus`, `fe`) and the other has none

Guardrail overrides are recorded as guardrail_variant_after_llm_yes when the LLM returns yes but a dangerous variant mismatch is still detected.

---

### 8. Grouping and Canonical Selection — `postprocessing.py`

Merge decisions are transitively closed using Union-Find (`build_groups()`). All listings connected by merge decisions form a single group.

Canonical title selection (`choose_canonical_title()`) scores each listing by:
- attribute completeness
- absence of promotional noise terms (`מבצע`, `sale`, `חדש`, `original`)
- title length
- price

Minimum price is computed after excluding suspicious low-price listings flagged by the pipeline, so extreme outliers do not affect the displayed group price.

---

## Screenshots

### Terminal — pipeline run

![Pipeline start and LLM decisions](screenshots/zsc3.png)

![Evaluation and production notes](screenshots/zsc2.png)

![LLM usage and error analysis](screenshots/zsc1.png)

### Output files — Excel view

**`llm_error_analysis.csv` — variant confusion cases (page 1)**

![Error analysis page 1](screenshots/excel1.png)

**`llm_error_analysis.csv` — variant confusion cases (page 2)**

![Error analysis page 2](screenshots/execl2.png)

**`llm_error_analysis.csv` — variant confusion cases (page 3)**

![Error analysis page 3](screenshots/execl3.png)

**`llm_error_analysis.csv` — variant confusion cases (page 4)**

![Error analysis page 4](screenshots/execl4.png)

**`llm_error_analysis.csv` — variant confusion cases (page 5)**

![Error analysis page 5](screenshots/execl5.png)

---

## Results

| Metric | Value |
|--------|-------|
| Input listings | 55 |
| Candidate pairs | 197 |
| Merged | 53 |
| Rejected | 144 |
| LLM calls | 36 (18.27% of pairs) |
| Avg LLM confidence | 0.883 |
| LLM vs rule disagreements | 13 |
| Product groups | 21 |

---

## Evaluation

| Metric | Score |
|--------|-------|
| Precision | 1.000 |
| Recall | 1.000 |
| Accuracy | 1.000 |
| False merge rate | 0.000 |
| TP / FP / TN / FN | 53 / 0 / 27 / 0 |

**Caveat:** gold labels were manually created by the pipeline author against the same dataset. These results validate that decisions are internally consistent and correct on known cases. A held-out blind evaluation on real Zap inventory would be required to confirm generalization.

---

## LLM Error Analysis

The 42 rows in llm_error_analysis.csv capture variant-related rejection cases, including LLM-reviewed rejections and post-LLM guardrail overrides.

Example high-frequency rejection reasons:
- display_variant_mismatch (e.g. Switch OLED vs standard)
- tier variant and color do not match
- tier variant does not match
- tier variant mismatch with llm_rule_disagreement
- tier variant mismatch between listings

Example: `Samsung Galaxy S23 256GB Black` vs `Samsung Galaxy S23 Ultra 256GB Black` — fuzzy score 0.949, full attribute agreement on every field except the Ultra tier variant. LLM correctly rejects. Final safety: guardrail_variant_after_llm_yes records cases where the LLM returned yes but the post-LLM variant guardrail still blocked the merge.

---

## Limitations and Production Path

**Attribute extraction** covers six manually defined product categories. At Zap's full catalog scale, this layer would need to be replaced with an LLM-based extractor or a fine-tuned NER model.

**Scoring weights** (50/35/15) were chosen based on signal reliability in this dataset and reviewed against the gold labels. A production system would tune weights per product category using a larger labeled corpus.

**Blocking** uses composite string keys. At very large scale, approximate nearest neighbor search (FAISS) over embeddings would catch pairs that share no overlapping string tokens.

**Evaluation** requires labels created independently of the pipeline author on a held-out set.

---

## Stack

| Component | Library |
|-----------|---------|
| Fuzzy matching | RapidFuzz |
| Embeddings | sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`) |
| LLM | OpenRouter API via openai-compatible client |
| Data processing | pandas |
| Clustering | Union-Find (custom implementation in `postprocessing.py`) |

> The `embeddings.position_ids UNEXPECTED` warning at model load is a known harmless artifact when loading BERT-family checkpoints via the sentence-transformers wrapper. It does not affect embedding output or quality.
