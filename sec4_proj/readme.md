# 📄 sec4_project_recommend_hotel_in_spain
- 호텔예약 및 리뷰 데이터로 해당데이터들을 통해 데이터를 통해 고객들이 원하는 조건의 호텔을 추천하는 모델을 제작했다.

- 데이터에서 몇몇 칼럼을 뽑아 해당 특성을 통해 고객이 원하는 유형의 호텔의 가장많은 숙박객수를 보유한 호텔을 알려주고, 고객이 원하는 호텔의 조건(싫어하는조건, 좋아하는조건)을 자연어 처리를 통해 가장 비슷한 리뷰를 가진 호텔 5개씩을 알려주는 모델이다.

## 📃 Data
### 데이터 내용
- 해당 데이터는 캐글의 [515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)
에서 받아왔습니다.

- 위 페이지에서 소개하기로는 Booking.com에서 스크랩 되었다고 하며, 유럽의 1493개의 럭셔리호텔들에 대한 515,000개의 리뷰를 데이터로 가지고 있다고 한다.

- 515,000개의 데이터 중 'Spain'의 'Barcelona'에 있는 호텔들의 2016~2017년 데이터로만 프로젝트를 진행하고자 한다.(60149개의 데이터)

### 데이터 선정이유
- 해당 데이터를 통해 고객들이 원하는 호텔은 추천하고자 한다.
  - 사용자는 호텔의 유형 즉 여행목적, 룸컨디션 등을 선택하고 자신이 찾고자 하는 호텔 혹은 가고싶지 않은 호텔의 조건을 추가적으로 기입하여 이전 리뷰에 해당 조건이 존재한다면, 제외하거나 추천하여 사용자들이 필요로 하는 호텔을 적극적으로 추천한다.

- "어떤 회사에서 높이 살 수 있을까?"
   - 해당 데이터가 '호텔'에 국한되어 있지만, 고객에게 제품을 판매하거나, 고객의 요구를 파악하여 추천하는 서비스를 제공하고자 하는 회사라면 비슷한 데이터를 가지고 있을경우 사용될 수 있을 것이라 생각한다. 

- "어디 회사의 어느부분에 적용해 볼 수 있을까"
  - "호텔예약서비스를 제공하는 회사의 호텔추천 서비스, 유통판매업의 같은 상품류 별 추천상품을 제공하는 서비스 등"

### 가설
- 호텔을 이용한 고객들의 리뷰를 분석하여 호텔추천 서비스를 이용하려는 고객들의 니즈에 맞춰서 호텔을 추천하여 고객의 편리성을 높힐 수 있을 것다
- 리뷰들을 자연어처리를 통해 분석하여 k-nn모델을 통해 가장 근접한 문서를 찾고 해당 문서에 해당하는 호텔의 이름을 뽑아 출력할 수 있을것이다.
- 추가적으로 서비스에 등록된 호텔 또한 회사의 고객이라 생각이 되기에, 각 호텔별 받은 리뷰의 상위 10개의 단어들을 제공할 수 도 있을 것이다.(일단은 전체호텔의 10개 단어만 추렸습니다.)

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
- 전체호텔들의 리뷰들을 분석하여 상위 10개의 단어를 선정함
  - Negative_Review
    <img width="351" alt="스크린샷 2021-07-01 오후 7 16 26" src="https://user-images.githubusercontent.com/73811590/124108080-d64e2480-daa0-11eb-9e19-3eb73d53b934.png">
  - Positive_Review
    <img width="361" alt="스크린샷 2021-07-01 오후 7 16 49" src="https://user-images.githubusercontent.com/73811590/124108127-e403aa00-daa0-11eb-9aec-29555a5dfbc3.png">
    
- 리뷰종류 별 총 단어의 개수
  - Negative_Review
    - ![image](https://user-images.githubusercontent.com/73811590/124109448-1a8df480-daa2-11eb-8af1-77e778df4084.png)
 
 
  - Positive_Review
    - ![image](https://user-images.githubusercontent.com/73811590/124109463-1e217b80-daa2-11eb-89a1-181f1b66b0d5.png)


## 🖥️ Model
-  Tf-IDF
```
# spacy tokenizer 함수
def tokenize(document):
    
    doc = nlp(document)
    # punctuations: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    return [token.lemma_.strip() for token in doc if (token.is_stop != True) and (token.is_punct != True) and (token.is_alpha == True)]
    
# Tf-IDF
N_tfidf = TfidfVectorizer(stop_words='english'
                        ,tokenizer=tokenize
                        ,ngram_range=(1,2)
                        ,max_df=.7
                        ,min_df=3
                        ,max_features = 20000
                       )
P_tfidf = TfidfVectorizer(stop_words='english'
                        ,tokenizer=tokenize
                        ,ngram_range=(1,2)
                        ,max_df=.7
                        ,min_df=3
                        ,max_features = 20000
                       )
Negative_Review_dtm = N_tfidf.fit_transform(Negative_Review)
Negative_Review_dtm = pd.DataFrame(Negative_Review_dtm.todense(), columns=N_tfidf.get_feature_names())
print(Negative_Review_dtm.head())

Positive_Review_dtm = P_tfidf.fit_transform(Positive_Review)
Positive_Review_dtm = pd.DataFrame(Positive_Review_dtm.todense(), columns=P_tfidf.get_feature_names())
print(Positive_Review_dtm.head())
```

- 유사도를 이용한 문서검색(NearestNeighbor (K-NN, K-최근접 이웃))
```
# NearestNeighbor (K-NN, K-최근접 이웃)
from sklearn.neighbors import NearestNeighbors

# dtm을 사용히 NN 모델을 학습시킵니다. (디폴트)최근접 5 이웃.
Negative_Review_nn = NearestNeighbors(n_neighbors=5, algorithm='kd_tree')
Negative_Review_nn.fit(Negative_Review_dtm)

Positive_Review_nn = NearestNeighbors(n_neighbors=5, algorithm='kd_tree')
Positive_Review_nn.fit(Positive_Review_dtm)
```

### 추가기능
```
# ['travel_perpos']을 통해서 해당 유형에서 가장 많이 예약된 상위 5개의 호텔 알려주기
def top5_hotel_by_travel_perpos(data, travel_perpos):
  travel_perpos_ls = data['travel_perpos'].unique()
  hotel_list = data['Hotel_Name'].unique()
  dic = {}
  for i in hotel_list:
    s = data[data['Hotel_Name']==i]['travel_perpos']=='travel_perpos'
    dic[i]=s.count()
  top5 = sorted(dic.items(), key=lambda x: x[1], reverse=True)[0:5]
  print(f"입력된 숙박목적은 '{travel_perpos}'이며, 해당 목적으로 가장많이 이용된 상위5개 호텔과 이용자 수 는 다음과 같습니다. ")
  return top5

# 사용자의 조건에 따라 호텔 추천하기
def kneighbors(data, Negative, Positive):
  hotel_list = data['Hotel_Name'].unique()
  test_N = N_tfidf.transform(Negative)
  test_P = P_tfidf.transform(Positive)
  Negative_kneighbors=Negative_Review_nn.kneighbors(test_N.todense())[1][0]
  Positive_kneighbors=Positive_Review_nn.kneighbors(test_P.todense())[1][0]
  print(f"입력된 싫어하는 호텔조건은\n {Negative}\n이며, 입력된 조건과 비슷한 리뷰가 있는 호텔은 다음과 같습니다.")
  for i in hotel_list[Negative_kneighbors]:
    print(i)
  print("\n")
  print(f"입력된 선호하는 호텔조건은\n {Positive}\n이며, 입력된 조건과 비슷한 리뷰가 있는 호텔은 다음과 같습니다.")
  for i in hotel_list[Positive_kneighbors]:
    print(i)
  return
```

# 모델 성능 확인
이용자1
- 여행목적: 'solotraveler'
- 싫어하는 호텔조건: 'bug, nosie, unkind staff'
- 좋아하는 호텔조건: 'nice step, good breakfast and pool'

- 조건 입력

```
user1 = {
    'travel_perpos':input("travel_perpos?:"),
    'n1':[input("negative_r?: ")],
    'p1':[input("positive_r?: ")]
    }
 >>> solotraveler
 >>> bug, nosie, unkind staff
 >>> nice step, good breakfast and pool
```

- 여행목적에 따른 상위 호텔 5개
```
top5_hotel_by_travel_perpos(df, user1['travel_perpos'])
```
 - 입력된 숙박목적은 'solotraveler'이며, 해당 목적으로 가장많이 이용된 상위5개 호텔과 이용자 수 는 다음과 같습니다. 
  - [('Eurostars Grand Marina Hotel GL', 1082),
  - ('Catalonia Atenas', 1061),
  - ('Catalonia Plaza Catalunya', 964),
  - ('Catalonia Barcelona Plaza', 932),
  - ('Barcelona Princess', 897)]
 
 
- 제시한 조건에 따른 추천호텔 및 비추천 호텔
```
kneighbors(df, user1['n1'], user1['p1'])
```
  - 입력된 싫어하는 호텔조건은
 ['bug, nosie, unkind staff']
이며, 입력된 조건과 비슷한 리뷰가 있는 호텔은 다음과 같습니다.
    - Catalonia Passeig de Gr cia 4 Sup
    - Eurohotel Diagonal Port
    - H10 Port Vell 4 Sup
    - Ilunion Bel Art
    - Hotel Omm


  - 입력된 선호하는 호텔조건은
 ['nice step, good breakfast, pool']
이며, 입력된 조건과 비슷한 리뷰가 있는 호텔은 다음과 같습니다.

    - Catalonia Barcelona Plaza
    - Catalonia Ramblas 4 Sup
    - Catalonia Park Putxet
    - Novotel Barcelona City
    - Grand Hotel Central
