# IndicCurate - Tamil Dataset Curation Pipeline

A complete, production-ready data curation pipeline for building high-quality Tamil-language training datasets. This project demonstrates an end-to-end workflow for data ingestion, cleaning, filtering, deduplication, quality assurance, and deployment to Hugging Face.

## 🎯 Project Overview

IndicCurate is a comprehensive solution for curating Tamil language datasets with multi-layer quality checks and automated processing. It handles both text and audio data with sophisticated filtering, deduplication, and validation mechanisms.

### Final Deliverables

- **Text Dataset**: 2,776 cleaned Tamil documents (14.1 MB)
- **Audio Dataset**: 100 Tamil audio samples with metadata (53.3 MB)
- **Deployed**: Available on [Hugging Face Hub](https://huggingface.co/datasets/KeerthanaNehru/indiccurate-tamil)

## ✨ Key Features

### Text Processing
- Multi-source ingestion (Wikipedia, OSCAR, news websites)
- Language identification and validation
- Length filtering (50-100,000 characters)
- Boilerplate and HTML removal
- PII redaction (phone numbers, emails)
- Exact, near-duplicate, and semantic deduplication
- Test set contamination detection

### Audio Processing
- Automatic download from OpenSLR SLR65
- Duration validation (0.5s - 120s)
- Signal-to-Noise Ratio (SNR) assessment
- Whisper speech-to-text WER comparison
- Speaker diversity analysis

### Quality Assurance
- Processing funnel tracking
- Data statistics and reporting
- Interactive Streamlit dashboard for manual review
- Multi-rater annotation support

## 📋 Dataset Statistics

### Text Corpus
| Metric | Value |
|--------|-------|
| Total Records | 2,776 |
| File Size | 14.1 MB |
| Language | Tamil (100%) |
| Avg Length | 5,808 characters |
| Quality | 91.1% survival rate |

### Audio Corpus
| Metric | Value |
|--------|-------|
| Total Samples | 100 |
| Duration | 0.16 hours (~10 minutes) |
| Format | WAV, 16kHz, mono |
| Speakers | 12 unique |
| QA Pass Rate | 87% |
| Avg SNR | 22.5 dB |

## 🚀 Quick Start

### Installation

1. **Clone the repository**
   ```bash
   cd /path/to/Data_Curator_Project/indiccurate
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys:
   # HF_TOKEN=your_huggingface_token
   # GROQ_API_KEY=your_groq_api_key
   ```

### Running the Pipeline

The entire pipeline can be run through shell scripts:

```bash
# Process text data through all phases
bash process_text.sh

# Process audio data
bash process_audio_phase6.sh

# Run Whisper audio quality checks
bash run_phase6_whisper.sh

# Launch the interactive dashboard
streamlit run src/annotation/app.py
```

## 📁 Project Structure

```
indiccurate/
├── src/
│   ├── ingestion/          # Data downloading & loading
│   ├── filtering/          # Quality filters
│   ├── dedup/             # Deduplication algorithms
│   ├── contamination/     # Test set overlap detection
│   ├── speech_qa/         # Audio quality assessment
│   ├── synthetic/         # Synthetic data generation
│   ├── annotation/        # Streamlit dashboard
│   └── common/            # Utilities & helpers
├── data/
│   ├── raw/              # Original downloaded data
│   ├── processed/        # Final cleaned parquet files
│   └── synthetic/        # Generated QA pairs
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables
└── README.md            # This file
```

## 🎛️ Dashboard Features

Launch the interactive dashboard to explore and annotate data:

```bash
streamlit run src/annotation/app.py
```

### Pages Available

1. **Processing Funnel** - Track records through each pipeline stage
2. **Data Browser** - Filter and explore text samples with metadata
3. **Manual Rating** - Rate samples on quality, toxicity, correctness
4. **Agreement Analysis** - Analyze inter-rater agreement metrics

## 📦 Deployment

Your datasets are automatically deployed to Hugging Face:

```bash
# View on Hugging Face
https://huggingface.co/datasets/KeerthanaNehru/indiccurate-tamil
```

### Loading Data

```python
# Using Hugging Face datasets library
from datasets import load_dataset

ds = load_dataset("KeerthanaNehru/indiccurate-tamil")

# Using pandas
import pandas as pd
text_df = pd.read_parquet("data/processed/text_clean.parquet")
audio_df = pd.read_parquet("data/processed/audio_clean.parquet")
```

## 📊 Quality Metrics

### Text Quality
- **PII Redacted**: 48 phone numbers, 3 emails
- **Duplicates Removed**: 257 (exact + near + semantic)
- **Contamination Filtered**: 3 records (0.11%)
- **Language Confidence**: 100% Tamil

### Audio Quality
- **SNR Average**: 22.5 dB (excellent)
- **Speaker Balance**: 50M/50F
- **QA Pass Rate**: 87%
- **Duration Range**: 2.3s - 12.5s

## 🔧 Configuration

Edit `.env` to customize:

```bash
# Hugging Face access
HF_TOKEN=hf_your_token_here

# Groq LLM API (for synthetic data)
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

## 📚 Dependencies

Key libraries used:
- `pandas` - Data manipulation
- `pyarrow` - Parquet file handling
- `scikit-learn` - Machine learning utilities
- `librosa` - Audio processing
- `streamlit` - Web dashboard
- `huggingface-hub` - Hugging Face integration
- `groq` - LLM API access

See `requirements.txt` for complete list.

## 🎓 Usage Examples

### Load and Explore Text Data

```python
import pandas as pd

df_text = pd.read_parquet("data/processed/text_clean.parquet")
print(f"Records: {len(df_text)}")
print(f"Languages: {df_text['lang_label'].unique()}")
print(f"Sources: {df_text['source'].unique()}")
```

### Filter and Sample

```python
# Get clean Tamil text only
clean_text = df_text[df_text['has_profanity'] == False]

# Random sample
sample = clean_text.sample(n=5)
for text in sample['text']:
    print(text[:100] + "...")
```

### Audio Access

```python
import librosa

df_audio = pd.read_parquet("data/processed/audio_clean.parquet")

# Load first audio file
audio_path = df_audio.iloc[0]['audio_path']
y, sr = librosa.load(audio_path, sr=16000, mono=True)
print(f"Loaded: {audio_path}, duration: {len(y)/sr:.2f}s")
```

## 📈 Performance

- **Total Pipeline Runtime**: ~2-3 hours (depending on data volume)
- **Text Processing**: Efficient batch processing with pandas
- **Audio Processing**: Parallel processing with librosa
- **Deduplication**: MinHash with 128 permutations
- **Deployment**: ~10 minutes for full upload to HF

## 🤝 Contributing

For suggestions or improvements, refer to the detailed workflow documentation.

## 📝 License

This project uses datasets with various licenses:
- Text: CC-BY-4.0, CC0, various
- Audio: OpenSLR SLR65 (research use)

## 📞 Support

For detailed workflow information, implementation details, and troubleshooting, see `DETAILED_WORKFLOW.md`.

## 🎉 Next Steps

1. Explore the dashboard: `streamlit run src/annotation/app.py`
2. Download data from Hugging Face
3. Fine-tune your own models
4. Contribute improvements

---

**Last Updated**: September 2026  
**Status**: ✅ Production Ready
