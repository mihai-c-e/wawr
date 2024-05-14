fact_extraction_prompt_1 = (
"""
Title: "Assessing Wikipedia-Based Cross-Language Retrieval Models".
Abstract: "This work compares concept models for cross-language retrieval: First, we adapt probabilistic Latent Semantic Analysis (pLSA) for multilingual documents. Experiments with different weighting schemes show that a weighting method favoring documents of similar length in both language sides gives best results. Considering that both monolingual and multilingual Latent Dirichlet Allocation (LDA) behave alike when applied for such documents, we use a training corpus built on Wikipedia where all documents are length-normalized and obtain improvements over previously reported scores for LDA. Another focus of our work is on model combination. For this end we include Explicit Semantic Analysis (ESA) in the experiments. We observe that ESA is not competitive with LDA in a query based retrieval task on CLEF 2000 data. The combination of machine translation with concept models increased performance by 21.1% map in comparison to machine translation alone. Machine translation relies on parallel corpora, which may not be available for many language pairs. We further explore how much cross-lingual information can be carried over by a specific information source in Wikipedia, namely linked text. The best results are obtained using a language modeling approach, entirely without information from parallel corpora. The need for smoothing raises interesting questions on soundness and efficiency. Link models capture only a certain kind of information and suggest weighting schemes to emphasize particular words. For a combined model, another interesting question is therefore how to integrate different weighting schemes. Using a very simple combination scheme, we obtain results that compare favorably to previously reported results on the CLEF 2000 dataset. "

Insights: ```json
[
{"type":"motivation", "text":"Compare concept models for cross-language retrieval.", "citation":"This work compares concept models for cross-language retrieval: First, we adapt probabilistic Latent Semantic Analysis (pLSA) for multilingual documents."},

{"type":"method", "text":"Adapting probabilistic Latent Semantic Analysis (pLSA) for multilingual documents in the context of cross-language retrieval", "citation":"First, we adapt probabilistic Latent Semantic Analysis (pLSA) for multilingual documents"},

{"type":"finding", "text":"In the context of adapting probabilistic Latent Semantic Analysis (pLSA) for multilingual documents and experimenting with different weighting schemes, a weighting favoring documents of similar lengths give the best results", "citation":"First, we adapt probabilistic Latent Semantic Analysis (pLSA) for multilingual documents. Experiments with different weighting schemes show that a weighting method favoring documents of similar length in both language sides gives best results."},

{"type":"finding", "text":"In the context of adapting probabilistic Latent Semantic Analysis (pLSA) for multilingual documents of similar lengths, both monolingual Latent Dirichlet analysis and multilingual Latent Dirichlet Analysis behave the same ", "citation":"Considering that both monolingual and multilingual Latent Dirichlet Allocation (LDA) behave alike when applied for such documents, "},

{"type":"method", "text":"In the context of adapting probabilistic Latent Semantic Analysis (pLSA) for multilingual documents, a training corpus of length-normalized Wikipedia documents produces improvements over previously reported scores.", "citation":"Considering that both monolingual and multilingual Latent Dirichlet Allocation (LDA) behave alike when applied for such documents, we use a training corpus built on Wikipedia where all documents are length-normalized and obtain improvements over previously reported scores for LDA. "},

{"type":"method", "text":"Exploring the integration of Explicit Semantic Analysis (ESA) with Latent Dirichlet Allocation (LDA) in cross-language retrieval tasks.", "citation":"Another focus of our work is on model combination. For this end we include Explicit Semantic Analysis (ESA) in the experiments."},

{"type":"finding", "text":"Explicit Semantic Analysis (ESA) is not competitive with Latent Dirichlet Allocation (LDA) in a query-based retrieval task using CLEF 2000 data.", "citation":"We observe that ESA is not competitive with LDA in a query based retrieval task on CLEF 2000 data."},

{"type":"finding", "text":"Combining machine translation with concept models results in a 21.1% improvement in mean average precision compared to using machine translation alone.", "citation":"The combination of machine translation with concept models increased performance by 21.1% map in comparison to machine translation alone."},

{"type":"challenge", "text":"Machine translation depends on parallel corpora, which may not be available for many language pairs, posing a challenge for cross-language information retrieval.", "citation":"Machine translation relies on parallel corpora, which may not be available for many language pairs."},

{"type":"method", "text":"Investigating the potential of Wikipedia's linked text to carry cross-lingual information without relying on parallel corpora.", "citation":"We further explore how much cross-lingual information can be carried over by a specific information source in Wikipedia, namely linked text."},

{"type":"finding", "text":"The best results in cross-language retrieval are achieved using a language modeling approach that does not depend on information from parallel corpora.", "citation":"The best results are obtained using a language modeling approach, entirely without information from parallel corpora."},

{"type":"challenge", "text":"The need for smoothing in language models raises questions about the soundness and efficiency of these models in cross-language retrieval.", "citation":"The need for smoothing raises interesting questions on soundness and efficiency."},

{"type":"finding", "text":"Link models in Wikipedia capture specific types of information, suggesting that different weighting schemes should be used to emphasize particular words.", "citation":"Link models capture only a certain kind of information and suggest weighting schemes to emphasize particular words."},

{"type":"method", "text":"Using a simple combination scheme to integrate different concept models and weighting schemes for improved cross-language retrieval performance.", "citation":"Using a very simple combination scheme, we obtain results that compare favorably to previously reported results on the CLEF 2000 dataset."}
]```

Extract a similar json for the following abstract. Be thorough and aim for at least 7 records. Use other type names if the extracted element does not match any type above.
Do not output the character " in your response, except for json formatting. Make sure to escape all characters such that the string can be parsed with the Python json library.

Title: "{{title}}"
Abstract: "{{abstract}}"
Insights:
""")


_fact_extraction_prompt_2 = ("You are given a research paper abstract on language models. Examine the abstract above"
                           " and create a comprehensive list of relevant facts, one sentence per fact, covering all "
                           "the background knowledge presented in the abstract, as well as the contribution of the "
                           "abstract . Assign a relevant type to each fact, such as: \"background\", \"goal\", "
                           "\"method\", \"result\", \"conclusion\" or others. Assign to each fact a word-for-word "
                           "comprehensive citation from the abstract that justifies the fact. The citation should "
                           "be large enough to provide context to the reader even without reading the rest of the "
                           "abstract. Answer in json format as described below. Answer with json and only json. "
                           "Escape characters appropriately so that your response parses as valid json when passed "
                           "to Python json.loads.  Make sure you do not output double quotes or the escape character "
                           "inside the strings. Only use double quotes for json formatting.\n\nOutput format:"
                           "[{\"type\":\"...\", \"fact\":\"...\", \"citation\": \"...\",},...]"
                           "\n\nTitle: \"{{title}}\"\n Abstract: \"{{abstract}}\""
                           )