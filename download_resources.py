# Author: Harsh Kohli
# Date Created: 29-09-2024

import os
from huggingface_hub import HfApi
from constants import LORA_MODULE_NAMES, ADAPTERS_DIR

api = HfApi()

for model in LORA_MODULE_NAMES:
    print("Downloading: " + model)
    api.snapshot_download(repo_id=model, repo_type="model", local_dir=os.path.join(ADAPTERS_DIR, model))

