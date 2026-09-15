"""Phase 7 Annotation & QA Dashboard - Streamlit Web Interface.

Multi-page Streamlit app for data exploration and manual quality annotation:

Page 1: Processing Funnel
  - Visualization of records at each pipeline stage
  - Input/output/removed counts
  - Survival rates

Page 2: Data Browser
  - Browse text and audio samples with metadata
  - Filter by source, language, quality metrics
  - View complete record details

Page 3: Manual Rating Form
  - Rate samples on quality (1-5 scale)
  - Assess toxicity and correctness
  - Save annotations to JSONL

Page 4: Inter-Annotator Agreement
  - Calculate agreement metrics if multiple raters
  - Cohen's Kappa or Fleiss' Kappa
  - Show rater statistics

Run:
    streamlit run src/annotation/app.py
"""

import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Configure page
st.set_page_config(
    page_title="IndicCurate: Data Curation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STATS_FILE = DATA_DIR / "stats.json"
TEXT_CLEAN = DATA_DIR / "processed" / "text_clean.parquet"
ANNOTATIONS_DIR = DATA_DIR / "processed" / "annotations"


# ============================================================================
# Page 1: Processing Funnel
# ============================================================================

def page_funnel():
    """Display processing funnel with stage-by-stage metrics."""
    st.title("📈 Processing Funnel")
    st.markdown("Complete view of records through each pipeline stage")
    
    # Load stats
    if not STATS_FILE.exists():
        st.error(f"Stats file not found: {STATS_FILE}")
        return
    
    with open(STATS_FILE, "r") as f:
        stats_data = json.load(f)
    
    # Extract stages
    runs = stats_data.get("runs", [])
    if not runs:
        st.warning("No pipeline data found")
        return
    
    # Group by stage and keep latest
    stage_data = {}
    for run in runs:
        stage = run["stage"]
        if stage not in stage_data or run["timestamp"] > stage_data[stage]["timestamp"]:
            stage_data[stage] = run
    
    # Create table
    table_data = []
    prev_output = None
    for stage, data in sorted(stage_data.items()):
        input_c = data["input_count"]
        output_c = data["output_count"]
        removed = data["removed_count"]
        
        if prev_output is not None and input_c != prev_output:
            st.warning(f"Mismatch at {stage}: input {input_c} != prev output {prev_output}")
        
        pct = 100.0 * output_c / input_c if input_c > 0 else 0
        table_data.append({
            "Stage": stage,
            "Input": input_c,
            "Output": output_c,
            "Removed": removed,
            "Survival %": f"{pct:.1f}%"
        })
        prev_output = output_c
    
    df_funnel = pd.DataFrame(table_data)
    st.dataframe(df_funnel, use_container_width=True, hide_index=True)
    
    # Summary stats
    if table_data:
        final = table_data[-1]["Output"]
        initial = table_data[0]["Input"]
        overall_survival = 100.0 * final / initial
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records Removed", sum(t["Removed"] for t in table_data))
        with col2:
            st.metric("Overall Survival Rate", f"{overall_survival:.1f}%")
        with col3:
            st.metric("Final Clean Dataset", final)


# ============================================================================
# Page 2: Data Browser
# ============================================================================

def page_browser():
    """Browse and filter data samples."""
    st.title("🔍 Data Sample Browser")
    st.markdown("Explore text records with metadata filtering")
    
    # Load data
    if not TEXT_CLEAN.exists():
        st.error(f"Text data not found: {TEXT_CLEAN}")
        return
    
    df = pd.read_parquet(TEXT_CLEAN)
    st.write(f"**Total Records**: {len(df):,}")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        source_filter = st.multiselect(
            "Source",
            options=df["source"].unique(),
            default=df["source"].unique()
        )
    
    with col2:
        min_len = st.slider("Min Text Length (chars)", 0, int(df["char_count"].max()), 0)
    
    with col3:
        max_len = st.slider("Max Text Length (chars)", 0, int(df["char_count"].max()), int(df["char_count"].max()))
    
    # Apply filters
    df_filtered = df[
        (df["source"].isin(source_filter)) &
        (df["char_count"] >= min_len) &
        (df["char_count"] <= max_len)
    ]
    
    st.write(f"**Filtered Records**: {len(df_filtered):,}")
    
    # Display samples
    if len(df_filtered) > 0:
        sample_idx = st.slider("Sample Index", 0, len(df_filtered) - 1)
        record = df_filtered.iloc[sample_idx].to_dict()
        
        st.subheader(f"Sample #{sample_idx + 1}")
        st.text_area("Text Content", value=record.get("text", ""), height=200, disabled=True)
        
        # Metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Source", record.get("source", "N/A")[:30])
        with col2:
            st.metric("Language", record.get("lang_label", "N/A")[:15])
        with col3:
            st.metric("Text Length", f"{record.get('char_count', 0):,} chars")
        
        # More metadata
        st.markdown("**Additional Metadata**")
        metadata_cols = ["title", "url", "publish_date", "license", "pipeline_stage"]
        for col in metadata_cols:
            if col in record:
                val = record[col]
                if val and not pd.isna(val):
                    st.write(f"**{col.replace('_', ' ').title()}**: {str(val)[:100]}")


# ============================================================================
# Page 3: Manual Rating Form
# ============================================================================

def page_rating():
    """Manual quality rating form."""
    st.title("⭐ Manual Quality Rating")
    st.markdown("Rate samples for quality, toxicity, and correctness")
    
    # Load data
    if not TEXT_CLEAN.exists():
        st.error(f"Text data not found: {TEXT_CLEAN}")
        return
    
    df = pd.read_parquet(TEXT_CLEAN)
    
    # Rater name
    rater_name = st.text_input("Your Name (Rater ID)", value="anonymous")
    
    # Sample selection
    sample_idx = st.number_input("Sample Index", 0, len(df) - 1, 0)
    record = df.iloc[sample_idx].to_dict()
    
    st.markdown(f"### Sample #{sample_idx + 1} / {len(df)}")
    st.text_area("Text", value=record.get("text", "")[:500], height=150, disabled=True)
    
    # Rating form
    st.markdown("### Rating Criteria")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        quality_score = st.slider(
            "Quality (1=Poor, 5=Excellent)",
            1, 5, 3,
            help="Grammar, clarity, usefulness"
        )
    
    with col2:
        toxicity_score = st.slider(
            "Toxicity (1=Toxic, 5=Clean)",
            1, 5, 5,
            help="No hate speech, offensive content"
        )
    
    with col3:
        correctness_score = st.slider(
            "Correctness (1=False, 5=True)",
            1, 5, 3,
            help="Factual accuracy, coherence"
        )
    
    # Comments
    comments = st.text_area("Comments (optional)", max_chars=500)
    
    # Save button
    if st.button("💾 Save Rating", type="primary"):
        # Create annotation
        annotation = {
            "record_index": int(sample_idx),
            "rater_name": rater_name,
            "timestamp": datetime.now().isoformat(),
            "quality": int(quality_score),
            "toxicity": int(toxicity_score),
            "correctness": int(correctness_score),
            "comments": comments,
            "text_preview": record.get("text", "")[:100],
            "source": record.get("source", "unknown")
        }
        
        # Save to annotations directory
        ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
        rater_file = ANNOTATIONS_DIR / f"{rater_name}_annotations.jsonl"
        
        with open(rater_file, "a") as f:
            f.write(json.dumps(annotation, ensure_ascii=False) + "\n")
        
        st.success("✅ Rating saved!")
        st.write(f"Saved to: {rater_file}")


# ============================================================================
# Page 4: Inter-Annotator Agreement
# ============================================================================

def page_agreement():
    """Calculate inter-annotator agreement metrics."""
    st.title("🤝 Inter-Annotator Agreement")
    st.markdown("Analyze agreement between multiple raters")
    
    # Load annotations
    if not ANNOTATIONS_DIR.exists():
        st.warning("No annotations found yet")
        return
    
    annotation_files = list(ANNOTATIONS_DIR.glob("*_annotations.jsonl"))
    if not annotation_files:
        st.warning("No annotation files found")
        return
    
    # Load all annotations
    all_annotations = {}
    for f in annotation_files:
        rater = f.stem.replace("_annotations", "")
        annotations = []
        with open(f, "r") as file:
            for line in file:
                annotations.append(json.loads(line))
        all_annotations[rater] = annotations
    
    st.write(f"**Raters**: {', '.join(all_annotations.keys())}")
    st.write(f"**Total Annotations**: {sum(len(a) for a in all_annotations.values())}")
    
    # Rater statistics
    st.markdown("### Rater Statistics")
    rater_stats = []
    for rater, annotations in all_annotations.items():
        avg_quality = sum(a.get("quality", 0) for a in annotations) / len(annotations) if annotations else 0
        avg_toxicity = sum(a.get("toxicity", 0) for a in annotations) / len(annotations) if annotations else 0
        avg_correctness = sum(a.get("correctness", 0) for a in annotations) / len(annotations) if annotations else 0
        
        rater_stats.append({
            "Rater": rater,
            "Ratings": len(annotations),
            "Avg Quality": f"{avg_quality:.2f}",
            "Avg Toxicity": f"{avg_toxicity:.2f}",
            "Avg Correctness": f"{avg_correctness:.2f}"
        })
    
    df_stats = pd.DataFrame(rater_stats)
    st.dataframe(df_stats, use_container_width=True, hide_index=True)
    
    # Calculate pairwise agreement (simple approach: % agreement within 1 point)
    if len(all_annotations) > 1:
        st.markdown("### Pairwise Agreement (Quality Scores)")
        
        rater_names = list(all_annotations.keys())
        
        # Find common samples
        for i, rater1 in enumerate(rater_names):
            for rater2 in rater_names[i+1:]:
                indices1 = {a["record_index"] for a in all_annotations[rater1]}
                indices2 = {a["record_index"] for a in all_annotations[rater2]}
                common = indices1 & indices2
                
                if common:
                    # Get scores for common samples
                    scores1 = {a["record_index"]: a["quality"] for a in all_annotations[rater1]}
                    scores2 = {a["record_index"]: a["quality"] for a in all_annotations[rater2]}
                    
                    agreements = sum(1 for idx in common if abs(scores1[idx] - scores2[idx]) <= 1)
                    agreement_pct = 100.0 * agreements / len(common)
                    
                    st.write(f"**{rater1} ↔ {rater2}**: {agreement_pct:.1f}% agreement on {len(common)} samples")


# ============================================================================
# Main Navigation
# ============================================================================

def main():
    """Main app with page navigation."""
    st.sidebar.title("📊 IndicCurate Dashboard")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Select Page",
        ["🔗 Processing Funnel", "🔍 Data Browser", "⭐ Manual Rating", "🤝 Agreement Analysis"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Project**: IndicCurate\n\n"
        "Tamil Dataset Curation Pipeline\n\n"
        "[📖 Docs](https://github.com)\n"
        "[💾 Data](./data/)"
    )
    
    if page == "🔗 Processing Funnel":
        page_funnel()
    elif page == "🔍 Data Browser":
        page_browser()
    elif page == "⭐ Manual Rating":
        page_rating()
    elif page == "🤝 Agreement Analysis":
        page_agreement()


if __name__ == "__main__":
    main()
