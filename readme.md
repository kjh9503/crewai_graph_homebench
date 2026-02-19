## 사용 방법
원래는 https://github.com/crewAIInc/crewAI 를 fork 한 뒤 graph memory 를 심어서 패키지화하려고 했는데,
현재 crewai 가 memory system 을 기존 short/long/entity 에서 통합 unified memory system 로 변경해서 
배포하는 packages 의 memory directory 는 최신기준으로는 short, long, entity 로 세분화 되어있지 않은 것 같습니다. 
대신, 기존 crewai package 가 있으시다면,

1. `crewai/` 폴더를 현재 crewai 프로젝트의 가상환경 경로로 복사합니다.  
   - 예: `.venv/lib/<python-version>/site-packages/`

2. `./data/` 폴더에서 `divide.py`를 실행해 입력 데이터를 분할합니다.
   ```bash
   cd data
   python divide.py
   cd ..
3.	전체 파이프라인을 실행합니다.
   `./run_all.sh`
