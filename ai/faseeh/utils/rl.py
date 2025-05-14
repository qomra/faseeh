# You may need to install the following libraries if not already available:
# pip install python-Levenshtein scikit-learn

import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def jaccard_similarity(text1: str, text2: str) -> float:
    """
    Returns the Jaccard similarity between two texts (token-based).
    Range: [0,1]
    """
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())

    if not tokens1 and not tokens2:
        return 1.0  # Both empty strings -> perfect similarity
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union)


def levenshtein_similarity(text1: str, text2: str) -> float:
    """
    Returns a normalized Levenshtein similarity between two strings, i.e.:
        1 - distance/max_len
    Range: [0,1]
    """
    dist = Levenshtein.distance(text1, text2)
    max_len = max(len(text1), len(text2))
    if max_len == 0:  # both empty
        return 1.0
    return 1 - (dist / max_len)


def tfidf_cosine_similarity(text1: str, text2: str) -> float:
    """
    Vectorizes both texts with TF-IDF and computes the cosine similarity.
    Range typically: [0,1], though negative values are rare in typical text.
    """
    if not text1.strip() and not text2.strip():
        return 1.0  # both empty
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return float(sim[0, 1])  # the off-diagonal element is text1 vs. text2


def token_f1_score(ref: str, hyp: str) -> float:
    """
    Computes a simple token-level F1 overlap between two strings.
    Used in QA tasks (similar to SQuAD's metric).
    Range: [0,1]
    """
    ref_tokens = ref.lower().split()
    hyp_tokens = hyp.lower().split()

    # Count overlap at the token level (multiset intersection)
    ref_count = {}
    for t in ref_tokens:
        ref_count[t] = ref_count.get(t, 0) + 1

    overlap = 0
    for t in hyp_tokens:
        if ref_count.get(t, 0) > 0:
            overlap += 1
            ref_count[t] -= 1

    precision = overlap / len(hyp_tokens) if hyp_tokens else 0
    recall = overlap / len(ref_tokens) if ref_tokens else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def combined_lexical_reward(text1: str, text2: str) -> float:
    """
    Computes four non-model-based similarity metrics and returns their average:
      1) Jaccard similarity
      2) Levenshtein similarity
      3) TF-IDF cosine similarity
      4) Token-level F1 overlap
    
    Returns a single reward value in [0,1].
    """
    jac = jaccard_similarity(text1, text2)
    lev = levenshtein_similarity(text1, text2)
    tfidf_sim = tfidf_cosine_similarity(text1, text2)
    f1 = token_f1_score(text1, text2)
    
    # Average them to get a final reward in [0,1]
    reward = (jac + lev + tfidf_sim + f1) / 4.0
    
    # Ensure it's within [0,1] due to any potential floating rounding
    reward = max(0.0, min(1.0, reward))
    return reward


# Example usage:
if __name__ == "__main__":
    textA = "أباة لغة في أباءة، وهي لغة الروس واللاجئ في بلاد العرب، وهي في معاجم اللغة بمعنى الأباءة، كما وردت في لسان العرب"
    textB = "الأباءة في لسان العرب هي أجمة القصب، والجمع أباء وهي من الجذر اللغوي أبأ "
    
    reward_value = combined_lexical_reward(textA, textB)
    print("Jaccard:", jaccard_similarity(textA, textB))
    print("Levenshtein:", levenshtein_similarity(textA, textB))
    print("TF-IDF Cosine:", tfidf_cosine_similarity(textA, textB))
    print("Token F1:", token_f1_score(textA, textB))
    print(f"Combined reward (avg): {reward_value:.4f}")
