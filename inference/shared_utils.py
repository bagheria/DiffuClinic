import os
import zipfile
import pandas as pd
from tqdm import tqdm


def load_data_from_zip(zip_path: str) -> pd.DataFrame:
    records = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        files = [f for f in z.namelist() if '/fulltext/' in f and f.endswith('.txt')]
        print(f"Found {len(files)} files in {zip_path}")

        for file in tqdm(files, desc="Loading data"):
            full_text = z.read(file).decode('utf-8')
            summary_file = file.replace('/fulltext/', '/summaries/').replace('.txt', '_sum.txt')
            summary_text = z.read(summary_file).decode('utf-8')
            records.append({'Full_Text': full_text, 'Summary': summary_text})

    print(f"Loaded {len(records)} samples")
    return pd.DataFrame(records)


def find_latest_checkpoint(output_dir: str, output_name: str = "checkpoint") -> tuple[int, list[dict]]:
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"checkpoint_{output_name}_"
    checkpoint_files = [f for f in os.listdir(output_dir) if f.startswith(prefix) and f.endswith('.csv')]
    if not checkpoint_files:
        return 0, []

    latest = sorted(checkpoint_files, key=lambda f: int(f.split('_')[-1].split('.')[0]))[-1]
    path = os.path.join(output_dir, latest)
    df = pd.read_csv(path)
    results = df.to_dict('records')
    print(f"Resuming from checkpoint: {len(results)} samples already done ({latest})")
    return len(results), results
