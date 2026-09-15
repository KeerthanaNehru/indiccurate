---
# Dataset Card for IndicCurate (Tamil)

tags:
- text-generation
- tamil
- lm-training
- data-curation
- low-resource

license: cc-by-sa-4.0
---

# IndicCurate Tamil Dataset

## Summary

**IndicCurate** is a curated Tamil-language text dataset designed for training Large Language Models (LLMs). It demonstrates a complete, reproducible pipeline for data curation in low-resource languages, combining quality filtering, deduplication, contamination detection, and manual QA.

**Final Size**: 2,776 records (91.1% survival from 3,047 raw records)

## Dataset Details

### Data Collection

The dataset combines three high-quality Tamil text sources:

1. **Tamil Wikipedia** (wikimedia/wikipedia:20231101.ta)
   - Source: Wikimedia Foundation
   - Records: 1,467 (52.8%)
   - Content: Encyclopedic articles on Tamil topics
   - License: CC-BY-SA-4.0

2. **OSCAR Tamil Subset** (AnanthZeke/oscar_tamil_2201)
   - Source: Hugging Face & Common Crawl
   - Records: 1,264 (45.4%)
   - Content: Web-crawled Tamil text
   - License: CC0

3. **Tamil News** (hindutamil.in)
   - Source: News website with permissive robots.txt
   - Records: 48 (1.7%)
   - Content: Recent Tamil news articles
   - License: Fair use (journalistic content)

### Data Processing Pipeline

The dataset underwent rigorous multi-stage processing:

| Stage | Method | Input | Output | Removed |
|-------|--------|-------|--------|---------|
| **Filtering** | Language ID (fastText), length (50-100K chars), HTML removal, PII redaction | 3,047 | 3,036 | 11 |
| **Exact Dedup** | SHA-256 content hashing | 3,036 | 2,962 | 74 |
| **Near-Dup Dedup** | MinHash + LSH (Jaccard ≥0.8) | 2,962 | 2,916 | 46 |
| **Semantic Dedup** | sentence-transformers + FAISS (cosine ≥0.99) | 2,916 | 2,779 | 137 |
| **Contamination** | n-gram & embedding overlap w/ test sets | 2,779 | 2,776 | 3 |
| **FINAL** | — | 3,047 | **2,776** | **271** |

### Quality Assurance

**Language & Content Quality:**
- Language Confidence: 1.00 (100% Tamil)
- PII Redacted: 46 phone numbers, 3 emails (no records removed)
- Profanity Flagged: 29 records (kept; for optional filtering)
- Contamination Rate: 0.11% (3/2,776 records)

**Text Statistics:**
- Median Length: 2,926 characters
- Mean Length: 5,808 characters
- Range: 50 to 100,000 characters

### Known Issues & Limitations

1. **Semantic Deduplication**
   - The multilingual MiniLM-L12-v2 model shows lower discriminative power for short Tamil texts (<500 chars)
   - Solution: Conservative threshold (0.99) minimizes false positives but may keep some near-duplicates
   - Recommendation: Fine-tune embeddings on Tamil corpus for future versions

2. **Data Size**
   - This dataset (2,776 records, ~8.5 MB) is designed for demonstration & research
   - For production LLM training, augment with additional Tamil sources or use data augmentation

3. **Bias & Coverage**
   - Sources skew towards Wikipedia (encyclopedic) and web-crawled content (informal)
   - Limited news coverage (1.7%)
   - No formal Tamil literature, technical documentation, or domain-specific text

4. **Language Variants**
   - Combines modern Tamil (Tamil Nadu, Sri Lanka, diaspora)
   - May include historical/classical Tamil in Wikipedia
   - No explicit dialect tags

## Dataset Structure

Each record is a JSON object with the following fields:

```json
{
  "text": "string",                    // Main content (50-100K chars)
  "source": "string",                // Origin (wikipedia, oscar, news)
  "license": "string",               // CC-BY-SA-4.0, CC0, or fair-use
  "ingestion_timestamp": "string",   // ISO 8601 UTC timestamp
  "pipeline_stage": "string",        // Current stage (raw→filtered→deduped→clean)
  
  // Optional metadata (depending on source):
  "title": "string",                 // Article/document title
  "url": "string",                   // Source URL
  "doc_id": "string",                // Source document ID
  "publish_date": "string",          // Publication date (news)
  "author": "string",                // Author (if available)
  "section": "string",               // Topic/section
  
  // Quality metrics:
  "lang_label": "string",            // Language label (__label__ta)
  "lang_confidence": float,          // fastText confidence [0-1]
  "char_count": int,                 // Text length
  "content_hash": "string",          // SHA-256 for dedup verification
  "pii_phone_redactions": int,       // Count of redacted phone numbers
  "pii_email_redactions": int,       // Count of redacted emails
  "has_profanity": bool,             // Profanity detected
  "profanity_matches": "array"       // Profanity terms found
}
```

## Intended Use

### Recommended Uses
- ✅ LLM pre-training corpus (supervised or unsupervised)
- ✅ Language model fine-tuning datasets
- ✅ NLP research & benchmarking
- ✅ Tamil language technology development
- ✅ Educational tools & resources
- ✅ Demonstrating data curation pipelines

### Not Recommended For
- ❌ Direct commercial products without further review
- ❌ Applications requiring domain-specific expertise (medical, legal)
- ❌ Real-time production systems (without additional QA)

## Curation Rationale

This dataset prioritizes **quality over quantity**:

1. **Multi-layer filtering** removes noise, HTML, non-Tamil content
2. **Deduplication** at three levels (exact, fuzzy, semantic) ensures diversity
3. **Contamination detection** prevents data leakage to eval benchmarks
4. **Manual QA dashboard** enables human-in-the-loop review
5. **Full reproducibility**: Open-source pipeline with all parameters documented

Trade-offs:
- 91.1% of raw data removed (aggressive quality threshold)
- Smaller final dataset (better suited to research than production)
- Processing time: ~50 minutes for text, ~5 hours for audio (with ML models)

## Ethical Considerations

### Dataset Provenance
- All sources obtained with respect to Terms of Service
- News content used under journalistic fair-use
- Wikipedia content retains CC-BY-SA-4.0 license

### Potential Biases
1. **Geographic**: Skews towards Indian Tamil (Tamil Nadu, Sri Lanka context)
2. **Socioeconomic**: Wikipedia users, web-content creators (higher education)
3. **Gender**: News & Wikipedia historically skew male
4. **Topic Coverage**: Encyclopedic bias (overrepresentation of academic topics)

### Mitigation Strategies
- Transparent documentation of sources & biases
- Reproducible pipeline for auditing & improvement
- Manual review tools (dashboard) for annotators
- Encouragement of community contributions

### Consent & Attribution
- Wikipedia: Community-contributed (CC-BY-SA requires attribution)
- OSCAR: Web-crawled (best-effort fair use)
- News: Original publications retain rights
- All uses should cite sources & dataset card

## Statistics

### Text Distribution
- **Unique Authors/Sources**: 3 (Wikipedia, OSCAR, News)
- **Language**: 100% Tamil (native script + Romanized/Tanglish minimal)
- **Median Record Age**: News (1-30 days), Wikipedia (varies, often months), OSCAR (web-crawled, date unknown)

### Processing Efficiency
- Phase 1 (Ingestion): 30 seconds
- Phase 2 (Filtering): 15 seconds
- Phase 3 (Deduplication): 50 minutes (semantic embedding compute-bound)
- Phase 4 (Contamination): 30 seconds
- Total: ~51 minutes for 3,047 → 2,776 records

## Additional Dataset Information

### Dataset Curators
- **Project**: IndicCurate (Tamil LLM Training Data Curation)
- **License**: MIT (pipeline code)
- **Citation**: [Provide citation if published]

### Collab Links
- GitHub Repository: [Link to repo]
- Hugging Face Model Card: [Link to HF]
- Paper (if applicable): [Link to paper]

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{indicurate_tamil_2026,
  title={IndicCurate Tamil: Curated Tamil Text Dataset for LLM Training},
  author={[Author Name]},
  year={2026},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/datasets/...}},
  doi={...}
}
```

## Changelog

- **v1.0** (2026-09-06): Initial release
  - 2,776 cleaned text records
  - Multi-layer deduplication & quality filtering
  - Full reproducible pipeline
  - Open-source annotation dashboard

## Related Datasets

- **IndicGLUE**: Indian languages NLU benchmarks
- **FLORES-200**: Multilingual machine translation (200 languages)
- **Tamil Wikipedia**: Wikimedia's Tamil encyclopedia
- **OSCAR**: Web-crawled corpus (220 languages)
- **CommonVoice**: Multilingual speech dataset

---

**Dataset Card Authors**: IndicCurate Team
**Last Updated**: 2026-09-06
