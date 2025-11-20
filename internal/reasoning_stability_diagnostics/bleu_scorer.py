
from typing import List
import sacrebleu

def bleu_score(predictions: List[str], golds: List[str], trg_lang: str) -> float:
    """Compute BLEU score using sacrebleu.
    Args:
        predictions (List[str]): List of hypothesis sentences.
        golds (List[str]): List of reference sentences.
        trg_lang (str): Target language code for sacrebleu.
    """

    assert len(predictions) == len(golds), "Hypothesis and references must have the same length. We don't support multiple references per hypothesis."
    
    hypotheses = predictions
    references = [golds]
    
    obj_bleu = sacrebleu.BLEU(trg_lang=trg_lang)
    return obj_bleu.corpus_score(hypotheses=hypotheses, references=references).score
