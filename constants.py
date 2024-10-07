# Author: Harsh Kohli
# Date Created: 29-09-2024

ADAPTERS_DIR = "adapters"
BASE_MODEL = "mistralai/Mistral-7B-v0.1"
MODEL_CONFIDENCE = "outputs/adapter_scores.json"
VALIDATION_CONFIDENCE = "outputs/validation_scores.json"

LORA_MODULE_NAMES = [
    "predibase/magicoder",
    "predibase/conllpp",
    "predibase/dbpedia",
    "predibase/cnn",
    "predibase/agnews_explained",
    "predibase/gsm8k",
    "predibase/customer_support",
    "predibase/glue_qnli",
    "predibase/glue_mnli",
    "predibase/glue_sst2",
    "predibase/glue_cola",
    "predibase/glue_stsb",
    "predibase/glue_mrpc",
    "predibase/glue_qqp",
    "predibase/tldr_headline_gen",
    "predibase/tldr_content_gen",
    "predibase/e2e_nlg",
    "predibase/wikisql",
    "predibase/hellaswag",
    "predibase/hellaswag_processed",
    "predibase/legal",
    "predibase/jigsaw",
    "predibase/bc5cdr",
    "predibase/covid",
    "predibase/drop",
    "predibase/drop_explained",
    "predibase/viggo"
]
