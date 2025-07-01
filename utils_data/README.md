## Generate Training Data

### Step 1: Create the environment
```
conda create -n make_data python=3.10
conda activate make_data
bash build.sh
```

### Step 2: Prepare CSV File
Create a CSV file listing the paths to ground-truth (GT) videos and their corresponding text descriptions. Use the following format:
```
path,text
/xxx/xxx/dog.mp4, A dog is sitting on the couch.
...
```

### Step 3: Configure Paths
Open `make_paired_data.sh` and modify the following variables:

- `INPUT_CSV`: Path to your CSV file
- `SAVE_PATH`: Directory to save the generated paired data

### Step 4: Run the Script
```
bash make_paired_data.sh
```

⚠️ **Notice:** The current version of `make_paired_data.sh` only supports `batch_size=1`.  
To process data in parallel, you can split the CSV file into multiple parts and run the script separately on each part.
