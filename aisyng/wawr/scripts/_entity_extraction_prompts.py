entity_extraction_prompt_1 = """
This is a paper abstract: "{{abstract}}"
List the entities from the following text, taken from the abstract above, together with their type: "{{text}}".
Answer in lower case and lemmatize entity names. Focus on entities that are relevand from a language model research and practical applications perspective.
Answer in json format like [{"name":"entity_name", "type": "entity_type"}, ...]. Answer with json and only json:
"""

entity_extraction_prompt_2 = """
This is a paper abstract: "{{abstract}}"
List the entities from the following text, taken from the abstract above: "{{text}}".
Use lower case for entity names. Extract only entities that are relevant from a language modeling research and practical applications perspective, such as models, algorithms, concepts, datasets, benchmarks, applications and others. Answer in lower case. Lemmatize entity names to their simplest singular form (for instance, "3d asset" instead of "3d assets").
Answer in json format like ["entity name 1", "entity name 2", ...]. Answer with json and only json:
"""