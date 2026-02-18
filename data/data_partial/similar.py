#!/usr/bin/env python3
"""
Find top 10 similar inputs to qnum 11 using BERTScore.
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple
from bert_score import score


def load_inputs(data_dir: Path, num_files: int = 100) -> Dict[int, str]:
    """Load input text from all JSON files."""
    inputs = {}
    for i in range(num_files):
        file_path = data_dir / f"generated_user_story_perceptions_{i}.json"
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                inputs[i] = data.get('input', '')
        else:
            print(f"Warning: {file_path} not found")
    return inputs


def find_top_k_similar(
    target_qnum: int,
    inputs: Dict[int, str],
    top_k: int = 10
) -> List[Tuple[int, float]]:
    """Find top-k most similar inputs to target_qnum using BERTScore."""
    
    if target_qnum not in inputs:
        raise ValueError(f"Target qnum {target_qnum} not found in inputs")
    
    target_input = inputs[target_qnum]
    
    # Prepare candidates (all inputs except target)
    candidates = []
    candidate_qnums = []
    
    for qnum in sorted(inputs.keys()):
        if qnum != target_qnum:
            candidates.append(inputs[qnum])
            candidate_qnums.append(qnum)
    
    print(f"Computing BERTScore similarity for qnum {target_qnum} against {len(candidates)} candidates...")
    
    # Compute BERTScore (all candidates vs target)
    references = [target_input] * len(candidates)
    P, R, F1 = score(candidates, references, lang='en', verbose=False)
    
    # Create list of (qnum, similarity_score)
    results = [(candidate_qnums[i], F1[i].item()) for i in range(len(candidates))]
    
    # Sort by similarity score (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results[:top_k]


def main():
    data_dir = Path(__file__).parent
    target_qnum = 11
    top_k = 10
    
    # Load all inputs
    print("Loading inputs...")
    inputs = load_inputs(data_dir)
    print(f"Loaded {len(inputs)} inputs\n")
    
    # Find top-k similar to target_qnum
    top_similar = find_top_k_similar(target_qnum, inputs, top_k)
    
    # Display results
    print("="*80)
    print(f"Top {top_k} inputs most similar to qnum {target_qnum}")
    print("="*80)
    print(f"\n[Target {target_qnum}]: {inputs[target_qnum]}\n")
    print("-"*80)
    
    for rank, (qnum, similarity) in enumerate(top_similar, 1):
        print(f"\n{rank}. qnum {qnum} - Similarity: {similarity:.4f}")
        print(f"   {inputs[qnum]}")
    
    print("\n" + "="*80)
    
    # Save results
    output_path = data_dir / f"similar_to_{target_qnum}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'target_qnum': target_qnum,
            'target_input': inputs[target_qnum],
            'top_similar': [
                {
                    'rank': rank,
                    'qnum': qnum,
                    'similarity': similarity,
                    'input': inputs[qnum]
                }
                for rank, (qnum, similarity) in enumerate(top_similar, 1)
            ]
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
