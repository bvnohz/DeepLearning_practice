

# Recurrent Neural Network

## Part 1 - Data Preprocessing

### Importing the libraries
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from google.colab import drive
drive.mount('/content/drive')

"""### Importing the training set"""

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

"""# 새 섹션

### Feature Scaling
"""

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

"""### 60 타임스텝과 1 출력을 갖는 데이터 구조를 생성

"""

X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

"""### Reshaping"""

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

"""## Part 2 - Building and Training the RNN

### Keras 라이브러리 임포트
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

"""### Initialising the RNN"""

regressor = Sequential()

"""### 첫 번째 LSTM 층과 일부 드롭아웃 정규화"""

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

"""### 두 번째 LSTM 층과 일부 드롭아웃 정규화"""

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

"""### 세 번째 LSTM 층과 일부 드롭아웃 정규화"""

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

"""### 네 번째 LSTM 층과 일부 드롭아웃 정규화"""

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

"""### output 레이어 추가"""

regressor.add(Dense(units = 1))

"""### RNN 컴파일"""

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')  # 손실함수, 최적화 알고리즘, 성능지표를 설정

"""### Traning set에 맞추기"""

regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

"""## Part 3 - Making the predictions and visualising the results

### 2017년의 실제 주가 가져오기
"""

dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

"""### 2017년 주가 예상"""

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
 #전체 데이터 중에서 데이터 세트와 그 전 60일 데이터 선택
inputs = inputs.reshape(-1,1) #inputs 배열의 형태를 조정하여 2D 배열로 변환
inputs = sc.transform(inputs)
X_test = [] #예측을 위한 테스트 데이터를 생성
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)  #훈련된 RNN 모델을 사용하여 테스트 데이터에 대한 주식 가격을 예측
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

"""### 결과 시각화"""

plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()