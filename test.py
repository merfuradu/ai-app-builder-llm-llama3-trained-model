from datasets import load_dataset

dataset = load_dataset("merfuradu/appbuilder", split="train")

def extract_inputs_outputs(dataset):
    inputs = []
    outputs = []

    for example in dataset['examples']:
        for item in example:
                input_text = item.get('input')
                output_text = item.get('output')

                if input_text is not None and output_text is not None:
                    inputs.append(input_text)
                    outputs.append(output_text)
                else:
                    print(f"Skipping example due to missing 'input' or 'output': {item}")

    return inputs, outputs

# print(extract_inputs_outputs(dataset)[0])

# Print out the first example to inspect the structure
print(dataset)
