import os
import cv2
import numpy as np

IMG_SIZE = 224

data = []
labels = []

classes = ["lung_opacity","viral_pneumonia","normal"]

for class_name in classes:

    path = os.path.join("dataset",class_name)

    label = classes.index(class_name)

    for img in os.listdir(path):

        img_path = os.path.join(path,img)

        image = cv2.imread(img_path)

        if image is None:
            print("Skipping:", img_path)
            continue

        image = cv2.resize(image,(IMG_SIZE,IMG_SIZE))

        data.append(image)
        labels.append(label)

data = np.array(data)/255.0
labels = np.array(labels)


import tensorflow as tf

def create_model():

    model = tf.keras.Sequential([

        tf.keras.layers.Conv2D(32,(3,3),activation="relu",
        input_shape=(224,224,3)),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64,(3,3),activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(128,(3,3),activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128,activation="relu"),

        tf.keras.layers.Dense(3,activation="softmax")
    ])

    model.compile(

        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


from sklearn.model_selection import KFold

k = 5

kfold = KFold(n_splits=k, shuffle=True)

scores = []

for train_idx, val_idx in kfold.split(data):

    X_train, X_val = data[train_idx], data[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]

    model = create_model()

    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_val,y_val)
    )

    loss,acc = model.evaluate(X_val,y_val)

    scores.append(acc)

print("Average accuracy:",np.mean(scores))


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt


y_bin = label_binarize(labels, classes=[0,1,2])

y_pred = model.predict(data)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(3):

    fpr[i], tpr[i], _ = roc_curve(y_bin[:,i], y_pred[:,i])

    roc_auc[i] = auc(fpr[i],tpr[i])


    plt.figure()

for i in range(3):

    plt.plot(
        fpr[i],
        tpr[i],
        label=f"{classes[i]} AUC = {roc_auc[i]:.2f}"
    )

plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()

model = create_model()

model.fit(data,labels,epochs=10,batch_size=32)

model.save("lung_xray_model.h5")