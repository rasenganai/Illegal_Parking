import keras as ks

def create_model(input_shape):
    base_model=ks.applications.VGG16(include_top=False,input_shape=input_shape)

    trainable=False
    for layer in base_model.layers:
        if layer.name == "block5_conv1":
            trainable=True
        layer.trainable=trainable

    model=ks.models.Sequential()
    model.add(base_model)
    model.add(ks.layers.Flatten())
    model.add(ks.layers.Dropout(rate=0.25))
    model.add(ks.layers.Dense(1024,activation="relu"))
    model.add(ks.layers.BatchNormalization())
    model.add(ks.layers.Dense(512,activation="relu"))
    model.add(ks.layers.BatchNormalization())
    model.add(ks.layers.Dense(128,activation="relu"))
    model.add(ks.layers.BatchNormalization())
    model.add(ks.layers.Dropout(rate=0.25))
    model.add(ks.layers.Dense(64,activation="relu"))
    model.add(ks.layers.BatchNormalization())
    model.add(ks.layers.Dense(1,activation="sigmoid"))
    opti=ks.optimizers.adamax(lr=0.0001)
    model.compile(optimizer=opti,loss=ks.losses.binary_crossentropy,metrics=["accuracy"])

    return model