from pathlib import Path

raw_file = Path("ml/data/raw/WISDM_ar_v1.1_raw.txt")
clean_file = Path("ml/data/raw/WISDM_clean.txt")

with raw_file.open("r") as infile, clean_file.open("w") as outfile:
    for line in infile:
        # Replace ",;" at the end with ";"
        if line.strip().endswith(",;"):
            line = line.replace(",;", ";")
        outfile.write(line)

print(f"Cleaned file written to {clean_file}")
