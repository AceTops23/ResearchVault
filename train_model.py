import spacy
from spacy.training.example import Example

# Load a pre-trained model or create a blank model
nlp = spacy.load("en_core_web_sm")
# OR: nlp = spacy.blank("en")

# Define your custom component (section detection) and add it to the pipeline
section_detection = nlp.create_pipe("ner")
nlp.add_pipe(section_detection)

# Load your annotated data (e.g., from a JSON file)
training_data = [
    ("Introduction is the first section.", {"entities": [(0, 12, "SECTION")]}),
    # Add more annotated examples
]

# Convert the training data into Example objects
examples = []
for text, annotations in training_data:
    example = Example.from_dict(nlp.make_doc(text), annotations)
    examples.append(example)

# Train the model
n_iter = 10  # Adjust the number of training iterations as needed
for _ in range(n_iter):
    for example in examples:
        nlp.update([example], losses={})

# Save the trained model
output_dir = "path_to_output_directory"
nlp.to_disk(output_dir)
