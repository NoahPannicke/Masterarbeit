import matplotlib
import pandas as pd
import numpy as np
import sklearn.metrics
from keras.layers import StringLookup
from keras.layers import BatchNormalization, Concatenate
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tensorflow.python.ops.init_ops_v2 import glorot_uniform
from keras.utils import to_categorical
from keras.losses import CategoricalCrossentropy
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
import tensorflow as tf
from keras.models import Model
from keras.metrics import Precision, Recall
from keras.layers import Dense, Input, Flatten, Dropout, Embedding
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping

classification_variable = "TARGET"
excluded_columns = ["TARGET"]


def drop_without_variance(df):
    variances = df.var()
    zero_var_cols = variances[variances == 0].index
    df = df.drop(zero_var_cols, axis=1)
    return df


def split_categorical_numerical(df):
    y = pd.DataFrame({classification_variable: df[classification_variable]})
    if(excluded_columns != ""): x = df.drop(excluded_columns, axis=1)
    else: x = df
    print(df.dtypes)
    categorical_vars = []
    numerical_vars = []
    for col in x.columns:
        if x[col].dtype == 'object':
            categorical_vars.append(col)
        else:
            numerical_vars.append(col)
    df_category = x[categorical_vars]
    df_numerical = x[numerical_vars]
    return x, y, categorical_vars, numerical_vars


def preprocess_data(df, categorical_vars):
    print("preprocess_data Start")
    for col in df.columns:
        dtype = df[col].dtype
        if dtype == 'object':
            df[col].fillna('fehlend', inplace=True)
        if dtype.name == 'category':
            df[col].cat.add_categories(['fehlend'], inplace=True)
            df[col].fillna('fehlend', inplace=True)
        elif dtype == 'float64' or dtype == 'int64':
            median = df[col].median()
            df[col].fillna(median, inplace=True)

    unique_values = {}
    for col in df[categorical_vars]:
        unique_values[col] = df[col].unique()

    value_to_int = {}
    for col in df[categorical_vars]:
        value_to_int[col] = {value: i for i, value in enumerate(unique_values[col])}

    for col in df[categorical_vars]:
        df[col] = df[col].map(value_to_int[col])
    return df



print("load_data start")
data = pd.read_csv('data/train.csv',encoding='unicode_escape')
X, Y, categorical_vars, numerical_vars = split_categorical_numerical(df = data)
X = preprocess_data(X, categorical_vars)
Y = preprocess_data(Y, [classification_variable])
print("load_data finish")

learning_rate=0.001
loss_str='CategoricalCrossentropy'
scaler_str='StandardScaler'
activation_funct='relu'
batch_size=30000
dropout_rate=0.2
n_hiddenlayers=4
n_neurons=100
optimizer_str='Adam'
print('Start Training')
X_train, X_val, y_train, y_val = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=42, shuffle=True)

print('Scaler start')
if(scaler_str == 'StandardScaler'):
    scaler = StandardScaler()
else:
    scaler = MinMaxScaler()
X_train[numerical_vars] = scaler.fit_transform(X_train[numerical_vars])
X_val[numerical_vars] = scaler.transform(X_val[numerical_vars])
print('Scaler fertig')

x_traindata = [X_train[numerical_vars]]
x_traindata = x_traindata + [pd.DataFrame(X_train[categorical_vars].iloc[:, i]) for i in
                             range(len(X_train[categorical_vars].columns))]
x_valdata = [X_val[numerical_vars]]
x_valdata = x_valdata + [pd.DataFrame(X_val[categorical_vars].iloc[:, i]) for i in
                         range(len(X_val[categorical_vars].columns))]

y_labels = Y[classification_variable].unique()

num_classes = len(y_labels)

y_train_oh = to_categorical(y_train, num_classes=num_classes)
y_val_oh = to_categorical(y_val, num_classes=num_classes)

input_layers = []
numerical_input = tf.keras.Input(shape=(X_train[numerical_vars].shape[-1],))
numerical_input = Flatten()(numerical_input)
input_layers.append(numerical_input)

unique_values = {}
for col in X_train[categorical_vars]:
    unique_values[col] = max(X_train[col])

embedding_layers = []

print(x_traindata)
print(x_valdata)

for col in X_train[categorical_vars]:
    num_unique_values = max(max(X_train[col]), max(X_val[col])) + 1
    embedding_size = min(10, (num_unique_values // 2) + 1)
    input_layer = Input(shape=(1,), name=col)
    input_layers.append(input_layer)
    embedding_layer = Embedding(input_dim=num_unique_values, output_dim=embedding_size, input_length=1)(input_layer)
    embedding_layer = Flatten()(embedding_layer)
    embedding_layers.append(embedding_layer)

if len(embedding_layers) > 0:
    category_embeddings = Concatenate()(embedding_layers)
    inputs = Concatenate(axis=-1)([category_embeddings, numerical_input])
else:
    inputs = Concatenate(axis=-1)([numerical_input])

layers = inputs
for i in range(n_hiddenlayers):
    layers = Dropout(rate=dropout_rate)(layers)
    layers = BatchNormalization()(layers)
    layers = Dense(n_neurons, activation=activation_funct, kernel_initializer=glorot_uniform())(layers)

output = Dense(num_classes, activation='softmax')(layers)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
if optimizer_str == 'Adam':
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
elif optimizer_str == 'SGD':
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
elif optimizer_str == 'RMSprop':
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
elif optimizer_str == 'Adadelta':
    optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
elif optimizer_str == 'Adamax':
    optimizer = tf.keras.optimizers.Adamax(learning_rate=learning_rate)
elif optimizer_str == 'Nadam':
    optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
elif optimizer_str == 'Adagrad':
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
elif optimizer_str == 'Ftrl':
    optimizer = tf.keras.optimizers.Ftrl(learning_rate=learning_rate)
else:
    raise ValueError(f"Unbekannter Optimizer: {optimizer_str}")

model = Model(inputs=input_layers, outputs=output)

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=optimizer,
              metrics=['accuracy', 'Recall', 'Precision'])

model.summary()

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0,
    patience=50,
    restore_best_weights=True,
)

model.fit(x=x_traindata, y=y_train_oh,
          epochs=25,
          batch_size=batch_size,
          verbose=1,
          callbacks=[early_stopping],
          validation_data=(x_valdata, y_val_oh)
          )

excluded_columns = ["DVER_PID","TARGET"]

data = pd.read_csv('data/test.csv',encoding='unicode_escape')
X, Y, categorical_vars, numerical_vars = split_categorical_numerical(df = data)
X = preprocess_data(X, categorical_vars)
Y = preprocess_data(Y, [classification_variable])

X[numerical_vars] = scaler.fit_transform(X[numerical_vars])

x = [X[numerical_vars]]
x = x + [pd.DataFrame(X[categorical_vars].iloc[:, i]) for i in
                             range(len(X[categorical_vars].columns))]

y_labels = Y[classification_variable].unique()

# Ensure that the number of labels is consistent
num_classes = len(y_labels)

y = to_categorical(Y, num_classes=num_classes)

pred_y = model.predict(x=x)

roc_auc_score(Y,pred_y[:,1])

classification_report(y_val, pred_y_test.argmax(axis=1))
cm = confusion_matrix(y_val, pred_y_test.argmax(axis=1), labels=y_labels.sort())

matplotlib.use('TkAgg')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y_labels)
disp.plot()
plt.show()

y_test = y_val
labels_train = np.unique(np.concatenate((y_train[classification_variable], pred_y_train.argmax(axis=1))))
labels_test = np.unique(np.concatenate((y_val[classification_variable], pred_y_test.argmax(axis=1))))

precision_train = precision_score(y_train, pred_y_train.argmax(axis=1), labels=labels_train, average='micro')
precision_test = precision_score(y_test, pred_y_test.argmax(axis=1), labels=labels_test, average='micro')
print("precision_train", precision_train)
print("precision_test", precision_test)

recall_train = recall_score(y_train, pred_y_train.argmax(axis=1), labels=labels_train, average='micro')
recall_test = recall_score(y_test, pred_y_test.argmax(axis=1), labels=labels_test, average='micro')
print("recall_train", recall_train)
print("recall_test", recall_test)

f1_train = f1_score(y_train, pred_y_train.argmax(axis=1), labels=labels_train, average='micro')
f1_test = f1_score(y_test, pred_y_test.argmax(axis=1), labels=labels_test, average='micro')
print("f1_train", f1_train)
print("f1_test", f1_test)

print("Finished Job")


