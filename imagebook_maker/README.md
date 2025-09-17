# 동화책 생성기 (Streamlit)

입력한 소설/책 내용을 동화 스타일의 장(챕터)으로 재구성하고, 각 장마다 1장의 이미지를 생성해주는 웹 앱입니다. Streamlit 기반으로 실행됩니다.

## 사전 준비
- Python 3.10+
- OpenAI API Key는 앱에서 나중에 입력해도 됩니다. (환경변수 설정도 가능)

## 설치
```bash
pip install -r requirements.txt
```

## 실행
```bash
streamlit run app.py
```

브라우저에서 열리는 UI의 사이드바에서 원하는 설정을 선택하고, 본문에 텍스트를 붙여 넣은 후 "챕터 생성 및 이미지 만들기"를 클릭하세요. OpenAI API Key는 사이드바에 비워둔 뒤, 필요 시 입력하면 됩니다.

## 주요 기능
- 입력 텍스트를 목표 연령/톤에 맞춰 동화책 챕터로 분할 및 다듬기
- 각 챕터당 1개의 일러스트 프롬프트 생성 및 이미지 생성
- 결과 미리보기, 개별 이미지 저장, 전체 ZIP 다운로드, JSON 구조 다운로드

## 환경변수로 API Key 설정 (선택)
PowerShell:
```powershell
$env:OPENAI_API_KEY = "sk-..."
```
cmd.exe:
```cmd
set OPENAI_API_KEY=sk-...
```

## 참고
- 입력이 너무 길면 모델 한계를 초과할 수 있습니다. 필요 시 요약 후 입력하세요.
- 생성된 이미지 및 텍스트는 배포 전 사람이 반드시 검수하세요.

## 라이선스
MIT
