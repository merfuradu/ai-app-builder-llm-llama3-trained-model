from langchain_community.llms import Ollama
from crewai import Agent, Task, Crew, Process
import wandb

wandb.login()

# Initialize a new wandb run
run = wandb.init(project="huggingface")

# Specify the path to save the downloaded model
save_path = "D:/PycharmProjects/openai/openai-env/artifacts/model2"

# Restore the model artifacts from wandb
artifact = run.use_artifact('merfuradu-ase/huggingface/model-hseq33ht:v0', type='model')
artifact_dir = artifact.download(save_path)


model = Ollama(
    model = "llama3",
    base_url = "http://localhost:11434")

professor_agent = Agent(role = "Application Development Consultant",
                      goal = """Provide detailed project proposals to clients based on their specific requirements, including technology stack, programming languages needed, development time, and cost estimates, and write it all in ROMANIAN.""",
                      backstory = """You are an experienced application development consultant, skilled in creating comprehensive project proposals for various types of applications.""",
                      allow_delegation = False,
                      verbose = True,
                      llm = model)

client_request = input("Solicitarea clientului: ")

task1 = Task(description=client_request,
             agent = professor_agent,
             expected_output="A detailed project proposal that describes in which programming languages should be the app, the estimated time of production and the estimated costs.")

crew = Crew(
            agents=[professor_agent],
            tasks=[task1],
            verbose=2
        )

result = crew.kickoff()

print(result)

wandb.finish()