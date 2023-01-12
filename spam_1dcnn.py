import numpy as np
import pandas as pd
import re
import json
import seaborn as sns
import matplotlib.pyplot as plt
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from collections import Counter
DATA_IN_PATH = './data/'
FONT_PATH = './font/'
stop_words = set(['은','는','이','가','하','아','것','들','의','있','되','수','보','등','한'])
##### 1. ham DATA  : for문으로 이쁘게 만들어 볼것 
# ham_data1 = pd.read_csv(DATA_IN_PATH+'ham_message_미용과건강.csv', encoding='utf8')
#ham_data2 = pd.read_csv(DATA_IN_PATH+'ham_message_상거래쇼핑.csv', encoding='utf8')
# ham_data3 = pd.read_csv(DATA_IN_PATH+'ham_message_시사교육.csv', encoding='utf8')
# ham_data4 = pd.read_csv(DATA_IN_PATH+'ham_message_식음료.csv', encoding='utf8')
# ham_data5 = pd.read_csv(DATA_IN_PATH+'ham_message_여가생활.csv', encoding='utf8')
#ham_data6 = pd.read_csv(DATA_IN_PATH+'ham_message_일과직업.csv', encoding='utf8')
ham_data7 = pd.read_csv(DATA_IN_PATH+'ham_message_주거와생활_old.csv', encoding='utf8')
# ham_data8 = pd.read_csv(DATA_IN_PATH+'ham_message_행사.csv', encoding='utf8')

#ham_data = pd.concat([ham_data2, ham_data6], axis=0, ignore_index=True)
# ham_data = pd.concat([ham_data1, ham_data2, ham_data3, ham_data4, ham_data5, ham_data6, ham_data7,ham_data8], axis=0, ignore_index=True)
ham_data =  ham_data7
print('총 샘플 수 : ', len(ham_data))   
print(ham_data[:5])
#%%

del ham_data['Unnamed: 0']
del ham_data['date']
#중복메세지 삭제
#print(ham_data[:5])  
ham_data.drop_duplicates(subset=['message'], inplace=True)  #113991
print(ham_data[:10])

#2 SPAM DATA
spam_data = pd.read_csv(DATA_IN_PATH+'KIS_DATA.csv', header=None, encoding='cp949')
spam_data = spam_data.drop(spam_data.iloc[:,0:10], axis=1)  # 불필요한 컬럼 삭제 
len(spam_data)
#train_data.info()  
#Data columns (total 1 columns):
#   Column  Non-Null Count    Dtype 
#---  ------  --------------    ----- 
# 0   10      1048574 non-null  object

spam_data.columns = ["message"]
spam_data['message'].nunique()   # 중복유무 확인 
spam_data.drop_duplicates(subset=["message"], inplace=True)  # 중복데이터 제거
spam_data['label'] = 1  # 라벨을 추가해줌
spam_data.head(5)  #14239 
len(spam_data)
# <class 'pandas.core.frame.DataFrame'>
# Int64Index: 14239 entries, 0 to 1048514
# Data columns (total 2 columns):
#  #   Column   Non-Null Count  Dtype 
# ---  ------   --------------  ----- 
#  0   message  14239 non-null  object
#  1   label    14239 non-null  int64 
# dtypes: int64(1), object(1)
# memory usage: 333.7+ KB

####3.ham,spam 데이터 합치기
raw_data = pd.concat([ham_data, spam_data], axis=0, ignore_index=True)
raw_data['label'].value_counts().plot(kind='bar')

print('spam:{}'.format(raw_data['label'].value_counts()[1]))
print('ham:{}'.format(raw_data['label'].value_counts()[0]))

X_data = raw_data['message']
Y_data = raw_data['label']

from sklearn.model_selection import train_test_split
X_train, X_test,  y_train, y_test  = train_test_split(X_data,Y_data, test_size=0.7, random_state=42, stratify=Y_data)
 
okt = Okt()

# X_train.info()
# y_train.info()

print(f'정상 메일의 비율 = {round(raw_data["label"].value_counts()[0]/len(raw_data) * 100,3)}%')
print(f'스팸 메일의 비율 = {round(raw_data["label"].value_counts()[1]/len(raw_data) * 100,3)}%')

clean_train_data =[]
clean_test_data=[]

def preproseesing(text, okt, remove_stopwords=False, stop_words=[]) :
    #한글추출
    message_text = re.sub("[^가-힣ㄱ-하-|\\s]","",text)
    message_text = re.sub('ㅠ', ' ', message_text)
    message_text = re.sub('ㅋ', ' ', message_text)
    message_text = ' '.join(message_text.split())
    
    word_text = okt.morphs(message_text,stem = True)  #형태소 단위로 나눈다
     
    #불용어 및 두글자 이상 명사 추출
    if remove_stopwords: 
        word  = [token for token in word_text if not token in stop_words and len(token) > 1 ]
    return word 

# trian 데이터 cleaning

for message in X_train :
    if type(message) ==str :
        clean_train_data.append(preproseesing(message, okt,remove_stopwords = True, stop_words = stop_words ))
    else :
        clean_train_data.append([])
        
print(clean_train_data[:5])


for message in X_test:
    if type(message) ==str :
        clean_test_data.append(preproseesing(message, okt,remove_stopwords = True, stop_words = stop_words ))
    else :
        clean_test_data.append([])
        
print(clean_test_data[:5])


print(f'train 정상 문자 = {round(y_train.value_counts()[0]/len(y_train) * 100,3)}%')
print(f'train 스팸 문자 = {round(y_train.value_counts()[1]/len(y_train) * 100,3)}%')
print(f'test 정상 문자 = {round(y_test.value_counts()[0]/len(y_test) * 100,3)}%')
print(f'test 스팸  = {round(y_test.value_counts()[1]/len(y_test) * 100,3)}%')

#%%

tokenizer = Tokenizer()
tokenizer.fit_on_texts(clean_train_data)
X_train_encoded = tokenizer.texts_to_sequences(clean_train_data) 
word_to_index = tokenizer.word_index
#print(word_to_index)
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(clean_raw_data)

threshold = 2
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

vocab_size = len(word_to_index) + 1
print('단어 집합의 크기: {}'.format((vocab_size)))
print('단어 집합의 크기 :',vocab_size)  #17974

print('문장의 최대 길이 : %d' % max(len(sample) for sample in X_train_encoded))
print('문장의  평균 길이 : %f' % (sum(map(len, X_train_encoded))/len(X_train_encoded)))
plt.hist([len(sample) for sample in X_data], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()
'''
단어 집합의 크기 : 51275
문자 최대 길이 : 953 
문자 평균 길이 : 5.4372767683069485   
'''
#%%
 
tokenizer = Tokenizer(vocab_size)
tokenizer.fit_on_texts(clean_train_data)
X_train_encoded = tokenizer.texts_to_sequences(clean_train_data)
# X_train_encoded = np.array(X_train_encoded)  #.astype array 로 변경 

# X_train_encoded = X_train_encoded.reshape(-1,1)
# print(len(X_train_encoded))
# print(len(y_train))

max_len = 200
#train set
X_train_padded = pad_sequences(X_train_encoded, maxlen = max_len)
#test set
X_test_encoded = tokenizer.texts_to_sequences(X_test)
X_test_padded = pad_sequences(X_test_encoded, maxlen = max_len)

print(X_train_padded.shape)
print(type(X_train_padded))

# from imblearn.over_sampling import RandomOverSampler
# oversample = RandomOverSampler(sampling_strategy=0.5)
# x_over, y_over = oversample.fit_resample(X_train_padded,y_train)

# X_train_padded = x_over   # Overfitting 후 훈련 

# print(x_over)

from imblearn.over_sampling import SMOTE
sm = SMOTE(sampling_strategy='minority')
X_resampled, y_resampled = sm.fit_resample(X_train_padded, y_train)

# print(X_resampled.shape, y_resampled.shape)

# type(y_resampled)
# print(y_resampled.value_counts()[0])
# print(y_resampled.value_counts()[1])
# print(f'test 정상 문자 = {round(y_resampled.value_counts()[0]/len(y_resampled) * 100,3)}%')
# print(f'test 스팸  = {round(y_resampled.value_counts()[1]/len(y_resampled) * 100,3)}%')
 

#%%
from tensorflow.keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Embedding,Dropout, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

model = Sequential()
model.add(Embedding(vocab_size, 32))
model.add(Dropout(0.2))
model.add(Conv1D(32, 5, strides=1, padding='valid', activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['acc'])  
#%%
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode = 'max', verbose=1, save_best_only=True)

#history = model.fit(X_train_padded, y_train, epochs = 10, batch_size=64, validation_split=0.2, callbacks=[es, mc])
history = model.fit(X_resampled, y_resampled, epochs = 10, batch_size=64, validation_split=0.2, callbacks=[es, mc])
#%%
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = range(1, len(loss)+1)

plt.plot(epochs, loss, 'b--', label = 'training loss')
plt.plot(epochs, val_loss, 'r:', label='validation loss')
plt.grid()
plt.legend()

plt.figure()
plt.plot(epochs, acc, 'b--', label = 'training accuracy')
plt.plot(epochs, val_acc, 'r:', label='validation accuracy')
plt.grid()
plt.legend()

plt.show()

#print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test_padded, y_test)[1]))

print("\n 테스트 정확도: %.4f" % (model.evaluate(X_resampled, y_resampled)[1]))

#인스타 스팸 데이터
#인스타그램 데이터
pred_data = pd.read_csv('./crawl/data/spam_insta.csv', header=None, encoding='utf-8')
pred_data.columns = ["message"]
pred_data['label'] = 1 
clean_pred_data = []

for message in pred_data["message"]:
    if type(message) ==str :
        clean_pred_data.append(preproseesing(message, okt,remove_stopwords = True, stop_words = stop_words ))
    else :
        clean_pred_data.append([])
        
print(clean_pred_data[:5])


tokenizer.fit_on_texts(clean_pred_data)
x_pred_encoded = tokenizer.texts_to_sequences(clean_pred_data) 
# word_to_index = tokenizer.word_index
vocab_size = word_to_index

print(vocab_size)
x_pred = pad_sequences(x_pred_encoded, maxlen=max_len)
y_pred = pred_data['label']

# def sentiment_predict(new_sentence):
#   new_sentence = re.sub(r'[^가-힣ㄱ-하-|\\s]','', new_sentence)
    
#   new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
#   new_sentence = [word for word in new_sentence if not word in stop_words] # 불용어 제거
#   encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
#   pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
#   score = float(model.predict(pad_new)) # 예측
#   print(score)
#   if(score > 0.5):
#     print("{:.2f}% 확률로 스팸문자 입니다.\n".format(score * 100))
#   else:
#     print("{:.2f}% 확률로 정상문자 입니다. \n".format((1 - score) * 100))

# sentiment_predict('대출')

#%%
insta_profile =[]
predict_score = []
point = []
m_type = []
for message in pred_data["message"]: 
  
        insta_sentence = preproseesing(message, okt,remove_stopwords = True, stop_words = stop_words )
        encoded = tokenizer.texts_to_sequences([insta_sentence]) # 정수 인코딩
        pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
        score = float(model.predict(pad_new)) # 예측
        insta_profile.append(message)
        predict_score.append(score)
    
        if(score > 0.5):
         # print("{:.2f}% 확률로 스팸문자 입니다.\n".format(score * 100))
           point.append('{:.2f}'.format(score * 100))
           m_type.append('스팸')
        else:
         # print("{:.2f}% 확률로 정상문자 입니다. \n".format((1 - score) * 100))
           point.append('{:.2f}'.format((1 - score) * 100))
           m_type.append('정상')
    
    # aa = sentiment_predict(pred_data) 
    # print(aa)
    # score = float(model.predict(pred_data)) # 예측
    
    # print('측정점수' , score)

print(insta_profile)
 
df = pd.DataFrame([insta_profile,predict_score,point,m_type]).T
df.columns=['프로필', '예측점수','point','구분']
 
# #MySQL에 저장
# # df.to_sql(name='prfDetail',con= conn, if_exists='replace', index='id', dtype= dtypesql )
# # df.to_sql(name='prfDetail',con= conn, if_exists='replace', index='id' )
# # conn.close()
# print("데이터베이스에 저장하였습니다.")
# # # # # # # csv 파일 생성
df.to_csv('result.csv')
