import itertools
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_wordnet_synonyms(term):
    """Gets synonyms for a term from WordNet, the original term is included."""
    synonyms = {term}
    for syn in wn.synsets(term):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace("_", " "))
    return list(synonyms)

def rank_candidates(base_term, candidates, top_k=3):
    """Ranks a list of candidate terms against a base term using semantic similarity."""

    base_emb = model.encode(base_term, convert_to_tensor=True)
    
    cand_embs = model.encode(candidates, convert_to_tensor=True)

    scores_tensor = util.cos_sim(base_emb, cand_embs)
    scores = scores_tensor[0].tolist()

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    
    print(f"Ranking (Frase, Score): {ranked}")

    return [s for s, _ in ranked[:top_k]]


def expand_terms_with_combinations(term:str, top_k:int=3):
    """
    Expands a single term by finding synonyms for each word, creating all possible combinations,
    and returning the top_k combinations that are most semantically similar to the original term.
    """

    words = term.lower().split()

    synonyms_per_word = [get_wordnet_synonyms(word) for word in words]

    candidate_tuples = itertools.product(*synonyms_per_word)
    candidate_terms = [" ".join(t) for t in candidate_tuples]
    print(f'Frases candidatas: {candidate_terms}')
    
    return rank_candidates(term, candidate_terms, top_k=top_k)

def build_search_string(terms_list, top_k=3):
    """
    Builds a search string from a list of terms.
    Each term is expanded into its most similar phrasal combinations.
    """
    clauses = []
    for term in terms_list:
        expanded_terms = expand_terms_with_combinations(term, top_k=top_k)
        
        quoted = [f'"{p}"' for p in expanded_terms]
        
        if quoted:
            clause = f"({' OR '.join(quoted)})"
            clauses.append(clause)
            
    return " AND ".join(clauses)

if __name__ == "__main__":
    # Write here the terms to expand
    terms = ["artificial intelligence", "healthcare"]
    
    search_string = build_search_string(terms, top_k=3)
    
    print(f"Query:{search_string}")
