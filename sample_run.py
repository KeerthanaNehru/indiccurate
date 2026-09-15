import pandas as pd

# Download and load directly
text_df = pd.read_parquet("https://huggingface.co/datasets/KeerthanaNehru/indiccurate-tamil/resolve/main/text_clean.parquet")
audio_df = pd.read_parquet("https://huggingface.co/datasets/KeerthanaNehru/indiccurate-tamil/resolve/main/audio_clean.parquet")

print(text_df.head())
print(audio_df.head())