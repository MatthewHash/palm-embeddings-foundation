import os
import numpy as np
import textwrap
import pandas as pd
import google.generativeai as palm

# Set API key
PALM_KEY = os.environ.get("PALM_API_KEY")
palm.configure(api_key=PALM_KEY)

# Create a variable to store the text
TEXT = "What is life?"

# List the models that support text embedding
models = [m for m in palm.list_models() if 'embedText' in m.supported_generation_methods]
model = models[0]

# CReate an embedding from the text
embedding = palm.generate_embeddings(model=model, text=TEXT)

# Print the text embedding.
print(embedding)
