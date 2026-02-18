#!/usr/bin/env python3
"""
F1 Score 기반 평가 스크립트
Gold: data/data_partial/generated_user_story_perceptions_{num}.json의 'input' 필드
Prediction: output/command_{num}.md의 'final_command' 필드
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter


def normalize_text(text: str) -> List[str]:
    """
    텍스트를 정규화하고 토큰화합니다.
    - 소문자 변환
    - 구두점 제거 및 공백으로 분리
    - 토큰 리스트 반환
    """
    text = text.lower()
    # 알파벳과 숫자만 남기고 나머지는 공백으로 치환
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    return tokens


def calculate_f1(gold_tokens: List[str], pred_tokens: List[str]) -> Tuple[float, float, float]:
    """
    Token 레벨에서 F1 score를 계산합니다.
    
    Returns:
        (precision, recall, f1): 각각의 점수
    """
    if not gold_tokens and not pred_tokens:
        return 1.0, 1.0, 1.0
    
    if not gold_tokens or not pred_tokens:
        return 0.0, 0.0, 0.0
    
    gold_counter = Counter(gold_tokens)
    pred_counter = Counter(pred_tokens)
    
    # 공통 토큰 개수 계산 (중복 고려)
    common_tokens = sum((gold_counter & pred_counter).values())
    
    # Precision: 예측한 토큰 중 gold에 있는 비율
    precision = common_tokens / len(pred_tokens) if pred_tokens else 0.0
    
    # Recall: gold 토큰 중 예측에 있는 비율
    recall = common_tokens / len(gold_tokens) if gold_tokens else 0.0
    
    # F1 Score
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    return precision, recall, f1


def load_gold(num: int, data_dir: Path) -> str:
    """
    Gold 데이터 로드
    """
    gold_path = data_dir / f"generated_user_story_perceptions_{num}.json"
    
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold 파일을 찾을 수 없습니다: {gold_path}")
    
    with open(gold_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get('input', '')


def load_prediction(num: int, output_dir: Path) -> str:
    """
    Prediction 데이터 로드
    """
    pred_path = output_dir / f"command_{num}.md"
    
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction 파일을 찾을 수 없습니다: {pred_path}")
    
    with open(pred_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Markdown 코드 블록에서 JSON 추출
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
    
    if not json_match:
        raise ValueError(f"JSON 블록을 찾을 수 없습니다: {pred_path}")
    
    json_str = json_match.group(1)
    data = json.loads(json_str)
    
    final_command = data.get('final_command', '')
    
    # final_command가 None이거나 dict인 경우 처리
    if final_command is None:
        return None
    if 'null' in final_command:
        return 'null'
    elif isinstance(final_command, dict):
        # dict인 경우 JSON 문자열로 변환하거나 적절한 문자열 표현 사용
        return json.dumps(final_command, ensure_ascii=False)
    elif isinstance(final_command, str):
        return final_command
    else:
        return str(final_command)


def evaluate_all(data_dir: Path, output_dir: Path) -> Dict:
    """
    모든 num에 대해 평가 수행
    """
    # 사용 가능한 num 찾기
    gold_files = sorted(data_dir.glob("generated_user_story_perceptions_*.json"))
    nums = []
    
    for gold_file in gold_files:
        match = re.search(r'generated_user_story_perceptions_(\d+)\.json', gold_file.name)
        if match:
            nums.append(int(match.group(1)))
    
    nums.sort()
    
    results = []
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    
    print(f"총 {len(nums)}개의 데이터를 평가합니다...\n")
    print(f"{'Num':<5} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 40)
    
    for num in range(56):
        try:
            # Gold와 Prediction 로드
            gold_text = load_gold(num, data_dir)
            pred_text = load_prediction(num, output_dir)
            if pred_text is None or pred_text == 'null':
                continue
            # 토큰화
            gold_tokens = normalize_text(gold_text)
            pred_tokens = normalize_text(pred_text)
            
            # F1 계산
            precision, recall, f1 = calculate_f1(gold_tokens, pred_tokens)
            
            results.append({
                'num': num,
                'gold': gold_text,
                'prediction': pred_text,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
            
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            
            print(f"{num:<5} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")
            
        except Exception as e:
            print(f"{num:<5} ERROR: {e}")
            continue
    
    # 평균 계산
    n = len(results)
    avg_precision = total_precision / n if n > 0 else 0.0
    avg_recall = total_recall / n if n > 0 else 0.0
    avg_f1 = total_f1 / n if n > 0 else 0.0
    
    print("-" * 40)
    print(f"{'AVG':<5} {avg_precision:<10.4f} {avg_recall:<10.4f} {avg_f1:<10.4f}")
    print()
    
    return {
        'results': results,
        'average': {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1
        },
        'total_count': n
    }


def save_detailed_results(results: Dict, output_path: Path):
    """
    상세 결과를 JSON 파일로 저장
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"상세 결과가 저장되었습니다: {output_path}")


def main():
    # 경로 설정
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data" / "data_partial"
    output_dir = base_dir / "output"
    
    # 평가 수행
    results = evaluate_all(data_dir, output_dir)
    
    # 상세 결과 저장
    result_path = base_dir / "evaluation_results.json"
    save_detailed_results(results, result_path)
    
    # 요약 출력
    print("\n=== 평가 요약 ===")
    print(f"평가된 데이터 수: {results['total_count']}")
    print(f"평균 Precision: {results['average']['precision']:.4f}")
    print(f"평균 Recall: {results['average']['recall']:.4f}")
    print(f"평균 F1 Score: {results['average']['f1']:.4f}")


if __name__ == "__main__":
    main()
