import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras

def preprocess_input(path):

    seed=2
    data=pd.read_csv(path)

    Y = data['Species']
    X = data.drop(['Id', 'Species'], axis=1)
    
    print("Shape of Input  features: {}".format(X.shape))
    print("Shape of Output features: {}".format(Y.shape))
    
    lbl_clf = LabelEncoder()
    Y_encoded = lbl_clf.fit_transform(Y)

    #Keras requires your output feature to be one-hot encoded values.
    Y_final = keras.utils.to_categorical(Y_encoded)

    print("Therefore, our final shape of output feature will be {}".format(Y_final.shape))

    x_train, x_test, y_train, y_test = train_test_split(X, Y_final, test_size=0.25, random_state=seed, stratify=Y_encoded, shuffle=True)

    print("Training Input shape\t: {}".format(x_train.shape))
    print("Testing Input shape\t: {}".format(x_test.shape))
    print("Training Output shape\t: {}".format(y_train.shape))
    print("Testing Output shape\t: {}".format(y_test.shape))

    std_clf = StandardScaler()
    x_train_new = std_clf.fit_transform(x_train)
    x_test_new = std_clf.transform(x_test)

    return x_train_new,x_test_new, y_train,y_test


def create_model(x_train, y_train):
    
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(10, input_dim=4, activation=tf.nn.relu))

    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(7, activation=tf.nn.relu))

    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(5, activation=tf.nn.relu))
    
    model.add(keras.layers.Dense(3, activation=tf.nn.softmax))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10, batch_size=7)

    return model

def save_model(model, version, save_path):

    model.export(f'{save_path}')
    

if __name__ == "__main__":

    # Get the current working directory
    parent_dir = os.getcwd()

    data_path=parent_dir+"\\data\\raw"
    train_path=data_path+"\\iris.csv"

    x_train, x_test, y_train, y_test = preprocess_input(train_path)

    # model = create_model(x_train, y_train)

    version = 1
    save_path=parent_dir+f"\\models\\iris_model\\{version}"

    # save_model(model, version, save_path)

    model_loaded = tf.saved_model.load(save_path)
    
    predictions=model_loaded.serve(x_test)

    print(predictions)