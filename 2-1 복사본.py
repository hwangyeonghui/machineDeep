fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

fish_data = [ [l,w] for l,w in zip(fish_length, fish_weight)]
fish_target = [1] * 35 + [0] * 14

from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()

# 넘파이 라이브러리 임포트
import numpy as np
# fish_data 와 fish_target 리스트를 넘파이 배열로 바꾸는 과정
input_arr = np.array(fish_data)
target_arr = np.array(fish_target)

#print(input_arr)
# 넘파이는 배열의 차원을 구분하기 쉽게 행과 열을 가지런히 출력함

#print(input_arr.shape)
# 넘파이는 배열의 크기를 알려주는 shape속성 제공 (샘플 수, 특성 수)

np.random.seed(42) 
# 넘파이에서 무작위 결과를 만드는 함수들은 실행할 때마다 다른 결과
#일정 결과 얻기 위해서는 초기에 랜덤 시드 지정

index = np.arange(49) 
#0-48까지 디폴트값 1씩 증가하는 배열 생성 -> 넘파이 arrange() 함수

np.random.shuffle(index) 
# index 값을 랜덤하게 섞는 과정

# 랜덤하게 섞인 index 배열을 사용해 전체 데이터를 훈련과 테스트 세트로 나눠보자
train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]

test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]

# 인덱스 값은 기억하고 있어야한다는 점을 기억해두자 
# 왜 인덱스 값 부터 랜덤하게 섞었는지 그 이유를 기억해두자

import matplotlib.pyplot as plt

#2차원 배열은 행과 열 인덱스를 콤마(,)로 나눠어 지정한다
plt.scatter(train_input[:,0], train_input[:, 1])
plt.scatter(test_input[:,0], test_input[:,1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()


kn = kn.fit(train_input, train_target)
score = kn.score(test_input, test_target)
print(score)

predict = kn.predict(test_input)
print(predict)
print(test_target)

