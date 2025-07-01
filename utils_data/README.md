## Generate Training Data

### Step 1: Prepare CSV File
Create a CSV file containing the paths to ground-truth (GT) videos and their corresponding text descriptions.

### Step 2: Configure Paths
Open `make_paired_data.sh` and modify the following variables:

- `INPUT_CSV`: Path to your CSV file
- `SAVE_PATH`: Directory to save the generated paired data

### Step 3: Run the Script
```
bash make_paired_data.sh
```

⚠️ **Notice:** The current version of `make_paired_data.sh` only supports `batch_size=1`.  
To process data in parallel, you can split the CSV file into multiple parts and run the script separately on each part.