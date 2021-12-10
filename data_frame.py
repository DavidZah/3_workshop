import pandas as pd
import openpyxl
import tensorflow as tf

from tensorflow.keras.layers import IntegerLookup
from tensorflow.keras.layers import Normalization
from tensorflow.keras.layers import StringLookup
import tensorflow.keras as keras
from tensorflow import keras
from tensorflow.keras import layers
# read 2nd sheet of an excel file
dataframe = pd.read_excel('Ai_inkubator.xlsx')

print(dataframe.head())

val_dataframe = dataframe.sample(frac=0.2, random_state=1337)
print(val_dataframe.index)
train_dataframe = dataframe.drop(val_dataframe.index)

print(
    "Using %d samples for training and %d for validation"
    % (len(train_dataframe), len(val_dataframe))
)

def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("Výsledek")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)

for x, y in train_ds.take(1):
    print("Input:", x)
    print("Target:", y)

train_ds = train_ds.batch(32)
val_ds = val_ds.batch(32)

def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature


def encode_categorical_feature(feature, name, dataset, is_string):
    lookup_class = StringLookup if is_string else IntegerLookup
    # Create a lookup layer which will turn strings into integer indices
    lookup = lookup_class(output_mode="binary")

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    lookup.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = lookup(feature)
    return encoded_feature

# Categorical features encoded as integers
sex = keras.Input(shape=(1,), name="sex", dtype="int64")
zajmi = keras.Input(shape=(1,), name="Zájmi", dtype="int64")

# Numerical features
age = keras.Input(shape=(1,), name="Věk")


all_inputs = [
    sex,
    zajmi,
    age
]

# Integer categorical features
sex_encoded = encode_categorical_feature(sex, "Pohlaví", train_ds, False)
zajmi_encoded = encode_categorical_feature(zajmi,"Zájmi",train_ds,False)

# Numerical features
age_encoded = encode_numerical_feature(age, "Věk", train_ds)


all_features = layers.concatenate(
    [
        sex_encoded,
        age_encoded,
        zajmi_encoded
    ]
)
x = layers.Dense(32, activation="relu")(all_features)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(all_inputs, output)
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

model.fit(train_ds, epochs=50, validation_data=val_ds)