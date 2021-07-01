# 📄 sec4_project_recommend_hotel_in_spain
- 호텔예약 및 리뷰 데이터로 해당데이터들을 통해 데이터를 통해 고객들이 원하는 조건의 호텔을 추천하는 모델을 제작했다.

- 데이터에서 몇몇 칼럼을 뽑아 해당 특성을 통해 고객이 원하는 유형의 호텔의 가장많은 숙박객수를 보유한 호텔을 알려주고, 고객이 원하는 호텔의 조건(싫어하는조건, 좋아하는조건)을 자연어 처리를 통해 가장 비슷한 리뷰를 가진 호텔 5개씩을 알려주는 모델이다.

## 📃 Data
- 해당 데이터는 캐글의 [515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)
에서 받아왔습니다.

- 위 페이지에서 소개하기로는 Booking.com에서 스크랩 되었다고 하며, 유럽의 1493개의 럭셔리호텔들에 대한 515,000개의 리뷰를 데이터로 가지고 있습니다.

- 515,000개의 데이터 중 
Spain의 Barcelona에 있는 호텔들의 2016~2017년 데이터로만 프로젝트를 진행함

## 🔠 Columns
- 특성설명
  - Hotel_Address: 호텔주소
  - Review_Date: 리뷰한 날짜
  - Average_Score: 호텔의 평균 점수(지난해 최신 코멘트를 기반으로 계산됨)
  - Hotel_Name: 호텔이름
  - Reviewer_Nationality: 리뷰어의 국적
  - Negative_Review: 부정적 리뷰. 없을 경우 'No Negative'로 표기
  - ReviewTotalNegativeWordCounts: 부정적 리뷰에 사용된 총 단어의 수
  - Positive_Review: 긍정적 리뷰. 없을 경우 'No Positive'로 표기
  - ReviewTotalPositiveWordCounts: 긍정적 리뷰에 사용된 총 단어의 수
  - Reviewer_Score: 리뷰 점수
  - TotalNumberofReviewsReviewerHasGiven: 리뷰어의 지난 리뷰들의 갯수
  - TotalNumberof_Reviews: 호텔이 가진 리뷰의 수
  - Tags: 리뷰어가 호텔에 단 태그
  - dayssincereview: 검토 날짜와 스크래핑 날짜 사이의 기간입니다.
  - AdditionalNumberof_Scoring: 투숙객들에 의한 호텔의 평가점수(리뷰를 작성하지 않은 평가자도 존재)
  - lat: 호텔의 위도
  - lng: 호텔의 경도


- 사용할 특성
  - Hotel_Address, Hotel_Name, Negative_Review, Positive_Review, Tags

## 전처리 과정
- 'Tags' 피처에서 숙박목적, 인원, 방규모 분리하기
  - 'Tags'특성을 소문자로 만들고 ```re.sub```을 사용한 전처리로 구분하기 편하게 만듬 
  - 인원, 방규모는 데이터가 너무 지저분 하여 제외함(추후 정제해서 조건으로 추가할 수 있도록 할 예정)


- 특성의 전처리: 소문자화, 알파벳과 숫자 ' , '빼고 삭제
  - ' , '기준으로 구분짓고 새로운 특성으로 생성 후 기존 'Tags'삭제


- 리뷰들의 토큰화
  - 소문자화 및 불용어처리
    - 불용어는 'spacy.load("en_core_web_sm")'를 사용했으나, 커스터마이징을 통해 더욱 정확성을 높힘
  - 표제어추출

- 같은 호텔리뷰 합쳐서 호텔들의 리뷰 리스트로 만들기
  - 같은호텔의 리뷰는 'Negative_Review', 'Positive_Review'로 자연어처리를 할 수 있도록 리스트화 함
  - 즉, 하나의 호텔은 각각 한 개의 'Negative_Review', 'Positive_Review'를 가짐

## 데이터 분석 

## 🖥️ Model
```
LabelEncoder() # 카테고리들의 범주화
Ensemble_pipe = make_pipeline(
    TargetEncoder(),
    SimpleImputer(),
    StandardScaler(), 
    RandomForestClassifier(random_state=2)
)
RandomizedSearchCV() # 최적의 파라미터값을 찾음
```


# 💻 Heroku
- 해당 레포의 헤로쿠 링크입니다.
- https://sec3-proj-tintin.herokuapp.com/

# 🏠 Home
- 해당 웹의 서비스를 설명하고, 사용할 수 있도록 만들어진 기본 페이지입니다.

# 🧍 User
- 사용자가 자신의 정보를 입력하여 데이터베이스에 저장하는 기능을 가진 페이지입니다.
- 사용자는 자신의 정보를 입력 후, 데이터베이스에 저장하여 Predict페이지에서 자신의 결과값을 확인할 수 있습니다.
- 결과값 확인후 User페이지로 돌아와서 자신의 정보를 삭제할 수 있습니다.

# 💳 Predict
- 사용자가 자신의 이름을 입력하여 데이터베이스의 정보를 불러온 후, 카드의종류를 예측하는 기능을 가능 페이지입니다.
- 사용자는 자신의 이름을 입력하고 자신이 발급받게될 카드의 종류의 예상, 즉 결과값을 확인할 수 있습니다.
- 이후 User페이지로 돌아가 자신의 정보를 삭제할 수 있습니다.

# 🖼️ Schema
- 해당 데이터베이스에서 사용되는 테이블의 스키마입니다.
<img width="359" alt="스크린샷 2021-06-02 오후 12 07 16" src="https://user-images.githubusercontent.com/73811590/120418226-14b9cc00-c39b-11eb-8bb7-6e6cf6360bb0.png">

