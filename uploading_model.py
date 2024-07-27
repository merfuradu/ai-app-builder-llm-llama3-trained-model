from huggingface_hub import HfApi, HfFolder, Repository, create_repo
import os
from dotenv import load_dotenv


# Define your Hugging Face username and repository name
username = "merfuradu"
repo_name = "trained_model"

# Local path to the model files
model_dir = "D:/PycharmProjects/openai/openai-env/artifacts/model-hseq33ht-v0"

# Create a repository on Hugging Face (if it doesn't already exist)
api = HfApi()
token = "HUGGING_FACE_TOKEN"
repo_id = f"{username}/{repo_name}"
create_repo(repo_id, exist_ok=True)


# Function to upload files
def upload_files_to_repo(repo_id, model_dir, token):
    api = HfApi()
    files = [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))]

    for file in files:
        file_path = os.path.join(model_dir, file)
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file,
            repo_id=repo_id,
            token=token
        )


upload_files_to_repo(repo_id, model_dir, token)
