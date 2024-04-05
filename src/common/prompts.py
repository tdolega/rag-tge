PROMPTS = {
    # my modification of ALCE
    1: """
Instruction: Write an accurate, engaging, and concise answer for the given question using only
the provided search results (some of which might be irrelevant) and cite them properly. Use an
unbiased and journalistic tone. Always cite for any factual claim at the end of each 
sentence using square brackets, for example: "Earth is the third planet from the Sun [2]. Sun is
the center of the Solar System [3].". When citing several search results, use [1][2][3].
Cite at least one document and at most three documents in each sentence. If multiple documents
support the sentence, only cite a minimum sufficient subset of the documents. Take a moment to
understand the context and then answer it with step-by-step reasoning in few sentences.
The answer should be concise and to the point.
""",
    # original ALCE
    2: """
Write an accurate, engaging, and concise answer for the given question using only
the provided search results (some of which might be irrelevant) and cite them properly. Use an
unbiased and journalistic tone. Always cite for any factual claim. When citing several search
results, use [1][2][3]. Cite at least one document and at most three documents in each sentence.
If multiple documents support the sentence, only cite a minimum sufficient subset of the
documents.
""",
    # new prompt
    3: """
Instruction: You are an expert Wikipedia editor. Provide an answer to the question posed,
using only the search results provided. Ignore any irrelevant search results. Each sentence
should contain one or more citations, referring to the facts in the search results. Place
the citations at the end of each sentence using square brackets like: [1][2]. 
""",
}
