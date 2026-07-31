from huggingface_hub import HfApi

REPO_ID = "NivedJk/muril-hinglish-sentiment"

api = HfApi()

print("Creating repository (if needed)...")

api.create_repo(
    repo_id=REPO_ID,
    repo_type="model",
    exist_ok=True,
)

print("Uploading model files...")

api.upload_folder(
    folder_path="models/muril_hinglish",
    repo_id=REPO_ID,
    repo_type="model",
)

print("\nUpload completed successfully!")
print(f"https://huggingface.co/{REPO_ID}")