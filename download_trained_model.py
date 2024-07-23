import wandb

# Login to WandB
wandb.login()


with wandb.init(project="huggingface"):
    # Download the model artifact
    artifact = wandb.use_artifact('model-hseq33ht:v0', type='model')
    artifact_dir = artifact.download()

print(f"Model downloaded to: {artifact_dir}")

