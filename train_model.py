print("TRAINING SCRIPT STARTED")

import os
import random
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ------------------------------------------------
# PATHS
# ------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "datasets", "CRICKET.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

print("Dataset path:", DATA_PATH)


# ------------------------------------------------
# REPRODUCIBILITY
# ------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset Loaded:", df.shape)


# ------------------------------------------------
# CLEANING
# ------------------------------------------------

df['extras_type'] = df['extras_type'].fillna('No_Extra')
df['dismissal_kind'] = df['dismissal_kind'].fillna('No_Dismissal')


# ------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------

df['ball_number'] = df['over'] * 6 + df['ball']

df = df.sort_values(['match_id', 'inning', 'ball_number'])

df['current_score'] = df.groupby(['match_id','inning'])['total_runs'].cumsum()

df['wickets_fallen'] = df.groupby(['match_id','inning'])['is_wicket'].cumsum()

TOTAL_OVERS = 20

df['overs_completed'] = df['ball_number'] / 6.0

df['run_rate'] = df['current_score'] / (df['overs_completed'] + 1e-6)

df['remaining_overs'] = TOTAL_OVERS - df['overs_completed']


# ------------------------------------------------
# TARGET: WIN / LOSS
# ------------------------------------------------

final_scores = df.groupby(['match_id','inning'])['total_runs'].sum().reset_index()

inning1 = final_scores[final_scores['inning']==1][['match_id','total_runs']]

inning2 = final_scores[final_scores['inning']==2][['match_id','total_runs']]

merged = inning1.merge(inning2, on='match_id', suffixes=('_1','_2'))

merged['inning2_win'] = (merged['total_runs_2'] > merged['total_runs_1']).astype(int)

df = df.merge(merged[['match_id','inning2_win','total_runs_1','total_runs_2']], on='match_id')

df['win'] = np.where(df['inning']==2, df['inning2_win'], 1 - df['inning2_win'])


# ------------------------------------------------
# TARGET: FINAL SCORE
# ------------------------------------------------

df['final_innings_score'] = np.where(

    df['inning']==1,

    df['total_runs_1'],

    df['total_runs_2']
)


# ------------------------------------------------
# SCORE BUCKET
# ------------------------------------------------

SCORE_BINS = [0,100,140,170,200,999]

SCORE_NAMES = [

    "Low (<100)",

    "Below Par (100-139)",

    "Par (140-169)",

    "Good (170-199)",

    "Excellent (200+)"
]

NUM_CLASSES = len(SCORE_NAMES)

df['score_bucket'] = pd.cut(

    df['final_innings_score'],

    bins=SCORE_BINS,

    labels=[0,1,2,3,4],

    right=False

).astype(int)


# ------------------------------------------------
# ENCODE TEAMS
# ------------------------------------------------

label_encoders = {}

for col in ['batting_team','bowling_team']:

    le = LabelEncoder()

    df[col] = le.fit_transform(df[col])

    label_encoders[col] = le


joblib.dump(label_encoders, os.path.join(MODEL_DIR,"label_encoders.pkl"))


# ------------------------------------------------
# FEATURES
# ------------------------------------------------

features = [

    'inning',
    'batting_team',
    'bowling_team',
    'ball_number',
    'current_score',
    'wickets_fallen',
    'run_rate',
    'remaining_overs'

]

X = df[features].values

y_win = df['win'].values

y_score = df['final_innings_score'].values

y_bucket = df['score_bucket'].values


# ------------------------------------------------
# SCALING
# ------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, os.path.join(MODEL_DIR,"feature_scaler.pkl"))


score_scaler = StandardScaler()

y_score_scaled = score_scaler.fit_transform(y_score.reshape(-1,1)).ravel()

joblib.dump(score_scaler, os.path.join(MODEL_DIR,"score_scaler.pkl"))


# ------------------------------------------------
# SEQUENCE GENERATION (FIXED)
# ------------------------------------------------

SEQUENCE_LENGTH = 20

X_seq = []
y_win_seq = []
y_score_seq = []
y_bucket_seq = []

for match_id in df['match_id'].unique():

    match_rows = df[df['match_id']==match_id]

    X_match = X_scaled[match_rows.index]

    y_win_match = y_win[match_rows.index]

    y_score_match = y_score_scaled[match_rows.index]

    y_bucket_match = y_bucket[match_rows.index]

    for i in range(len(match_rows)-SEQUENCE_LENGTH):

        X_seq.append(X_match[i:i+SEQUENCE_LENGTH])

        y_win_seq.append(y_win_match[i+SEQUENCE_LENGTH-1])

        y_score_seq.append(y_score_match[i+SEQUENCE_LENGTH-1])

        y_bucket_seq.append(y_bucket_match[i+SEQUENCE_LENGTH-1])


X_lstm = np.array(X_seq)

y_win_arr = np.array(y_win_seq)

y_score_arr = np.array(y_score_seq)

y_bucket_arr = np.array(y_bucket_seq)


print("Sequence shape:",X_lstm.shape)


# ------------------------------------------------
# ONE HOT
# ------------------------------------------------

y_bucket_onehot = tf.keras.utils.to_categorical(y_bucket_arr, NUM_CLASSES)


# ------------------------------------------------
# TRAIN TEST SPLIT
# ------------------------------------------------

X_train,X_test,yw_train,yw_test,ys_train,ys_test,yb_train,yb_test = train_test_split(

    X_lstm,

    y_win_arr,

    y_score_arr,

    y_bucket_onehot,

    test_size=0.2,

    random_state=SEED

)


# ------------------------------------------------
# CLASS WEIGHTS
# ------------------------------------------------

cw = compute_class_weight(

    'balanced',

    classes=np.array([0,1]),

    y=yw_train

)

class_weight = {0:cw[0],1:cw[1]}


# ------------------------------------------------
# MODEL
# ------------------------------------------------

inputs = Input(shape=(SEQUENCE_LENGTH,8))

x = Bidirectional(LSTM(128, return_sequences=True))(inputs)

x = BatchNormalization()(x)

x = Dropout(0.3)(x)

x = Bidirectional(LSTM(64))(x)

x = BatchNormalization()(x)

x = Dropout(0.2)(x)


shared = Dense(64,activation='relu')(x)


win_out = Dense(1,activation='sigmoid',name='win_output')(shared)

score_out = Dense(1,activation='linear',name='score_output')(shared)

bucket_out = Dense(NUM_CLASSES,activation='softmax',name='bucket_output')(shared)


model = Model(inputs=inputs, outputs=[win_out,score_out,bucket_out])


model.compile(

    optimizer=Adam(0.0003),

    loss={

        'win_output':'binary_crossentropy',

        'score_output':'mse',

        'bucket_output':'categorical_crossentropy'

    },

    metrics={

        'win_output':'accuracy',

        'score_output':'mae',

        'bucket_output':'accuracy'

    }

)


model.summary()


# ------------------------------------------------
# TRAIN
# ------------------------------------------------

model.fit(

    X_train,

    {

        'win_output':yw_train,

        'score_output':ys_train,

        'bucket_output':yb_train

    },

    epochs=20,

    batch_size=32,

    validation_split=0.1,

    callbacks=[

        EarlyStopping(patience=5, restore_best_weights=True),

        ReduceLROnPlateau(patience=3)

    ]

)


# ------------------------------------------------
# EVALUATION
# ------------------------------------------------

pred_win,pred_score,pred_bucket = model.predict(X_test)

pred_win = (pred_win>0.5).astype(int)

acc = accuracy_score(yw_test,pred_win)

print("Win Accuracy:",acc)


# ------------------------------------------------
# SAVE MODEL
# ------------------------------------------------

model.save(os.path.join(MODEL_DIR,"bilstm_multioutput_model.h5"))

print("MODEL SAVED")


# ------------------------------------------------
# SAVE SCORE META
# ------------------------------------------------

joblib.dump({

    "bins":SCORE_BINS,

    "names":SCORE_NAMES,

    "num_classes":NUM_CLASSES

},os.path.join(MODEL_DIR,"score_bucket_meta.pkl"))


print("ALL FILES SAVED")