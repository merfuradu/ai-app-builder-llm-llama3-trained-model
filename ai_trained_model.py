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

professor_agent = Agent(role = "Consultant în Dezvoltare Aplicații",
                      goal = """Furnizați propuneri detaliate pentru proiecte clienților, bazate pe cerințele specifice ale acestora, inclusiv tehnologia necesară, limbajele de programare, timpul estimat de dezvoltare și estimările de costuri. În propunere, trebuie să includeți un plan detaliat care să acopere etapele esențiale ale proiectului, și să oferiți estimări exacte pentru costuri și ore de muncă. Toate informațiile trebuie să fie prezentate în limba română, într-un format profesional și clar.""",
                      backstory = """Sunteți un consultant experimentat în dezvoltarea aplicațiilor, specializat în crearea de propuneri cuprinzătoare pentru diverse tipuri de aplicații. Aveți competențe avansate în evaluarea cerințelor clienților și în elaborarea unor estimări precise pentru costuri și timp de dezvoltare, și sunteți capabil să redați toate informațiile într-un format bine structurat și în limba română.""",
                      allow_delegation = False,
                      verbose = True,
                      llm = model)

def generate_proposal(client_request):
    task1 = Task(description=client_request,
                agent = professor_agent,
                expected_output="Această ofertă preliminară trebuie să includă următoarele informații: 1. Introducere și Context: O scurtă introducere care explică că oferta se bazează pe informațiile furnizate și că înainte de a începe dezvoltarea aplicației este necesară parcurgerea unor etape de planificare esențiale. 2. Etape de Planificare: Diagramă Logică: Descrie procesul de creare a unei diagrame logice pentru arhitectura aplicației. Menționează că această etapă definește structura și fluxul de date al aplicației. Diagramă ER: Explică realizarea unei diagrame entitate-relație (ER) pentru a structura baza de date. Sublinează importanța acestei diagrame în organizarea și gestionarea datelor. Design în Figma: Precizează că se va crea un design inițial în Figma pentru a clarifica aspectul și funcționalitatea interfeței utilizatorului. 3. Costuri și Contract: Diagrama Logică și ER: Costul estimat și TVA-ul pentru aceste documente. Design în Figma: Costul estimat și TVA-ul pentru designul interfeței. Contract și Plată: Menționează că înainte de a începe dezvoltarea, va trebui să fie semnat un contract și să fie achitată suma în avans pentru etapele de planificare. 4. Estimare Finală: Oferta Finală: Clarifică că oferta finală va fi ajustată pe baza etapei de planificare, și că aceasta include costuri și timp de livrare definite după finalizarea planificării. 5. Funcționalități Incluse: Descriere generală a funcționalităților care vor fi incluse în aplicație. 6. Estimare de Cost și Timp: Cost Total: Estimarea costului total pentru dezvoltarea aplicației, inclusiv TVA. Timp de Livrare: Estimarea timpului necesar pentru finalizarea dezvoltării, bazat pe complexitatea funcționalităților și resursele disponibile. Această ofertă este orientativă și poate fi ajustată în funcție de cerințele suplimentare sau modificările de specificație. Timpul și costurile finale vor fi stabilite în urma finalizării etapei de planificare.")

    crew = Crew(
                agents=[professor_agent],
                tasks=[task1],
                verbose=2
            )

    result = crew.kickoff()
    return result

wandb.finish()