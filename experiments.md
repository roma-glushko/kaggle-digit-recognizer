## CNN

### Baseline CNN

- Best Val Loss: 0.0337
- Best Val Accuracy: 0.9910
- 0.04179650917649269, 0.9896825551986694

```
cnn_input_layer = Input((28, 28, 1))

cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn_input_layer)
cnn = MaxPool2D(2)(cnn)

cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = MaxPool2D(2)(cnn)

cnn = Flatten()(cnn)


cnn = Dense(128)(cnn)
cnn = BatchNormalization()(cnn)
cnn = ReLU()(cnn)
cnn = Dropout(0.2)(cnn)

cnn = Dense(64)(cnn)
cnn = BatchNormalization()(cnn)
cnn = ReLU()(cnn)
cnn = Dropout(0.2)(cnn)

cnn_output_layer = Dense(NUM_CLASSES, activation = 'softmax')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

optimizer = Nadam(lr=0.0005)
batch_size = 32
```
### CNN

- Best Val Loss: 0.0369
- Best Val Accuracy: 0.9904
- 0.03625720739364624, 0.9896825551986694

```
cnn_input_layer = Input((28, 28, 1))

cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn_input_layer)
cnn = MaxPool2D(2)(cnn)

cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = MaxPool2D(2)(cnn)

cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)

cnn = Flatten()(cnn)

cnn = Dense(64)(cnn)
cnn = BatchNormalization()(cnn)
cnn = ReLU()(cnn)
cnn = Dropout(0.2)(cnn)

cnn_output_layer = Dense(NUM_CLASSES, activation = 'softmax')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

optimizer = Nadam(lr=0.0005)
batch_size = 64
```

### CNN

- Best Val Loss: 0.0375
- Best Val Accuracy: 0.9902
- Test: 0.037853945046663284, 0.9896825551986694
- Kaggle: 0.99050

```
cnn_input_layer = Input((28, 28, 1))

cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn_input_layer)
cnn = MaxPool2D(2)(cnn)

cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = MaxPool2D(2)(cnn)

cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)

cnn = Flatten()(cnn)

cnn = Dense(64)(cnn)
cnn = BatchNormalization()(cnn)
cnn = ReLU()(cnn)
cnn = Dropout(0.2)(cnn)

cnn_output_layer = Dense(NUM_CLASSES, activation = 'softmax')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

optimizer = Nadam(lr=0.002)
batch_size=128,
```

### CNN

- Best Val Loss: 0.0331
- Best Val Accuracy: 0.9902
- 0.03528701514005661, 0.9892857074737549

```
cnn_input_layer = Input((28, 28, 1))

cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn_input_layer)
cnn = MaxPool2D(2)(cnn)

cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = MaxPool2D(2)(cnn)

cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)

cnn = Flatten()(cnn)

cnn = Dense(64)(cnn)
cnn = BatchNormalization()(cnn)
cnn = ReLU()(cnn)
cnn = Dropout(0.2)(cnn)

cnn_output_layer = Dense(NUM_CLASSES, activation = 'softmax')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

optimizer = Nadam(lr=0.0005)
batch_size=128,
```

### CNN (HeNorm Init)

- Best Val Loss: 0.0321
- Best Val Accuracy: 0.9910
- 0.039722077548503876, 0.9900793433189392

```
cnn_input_layer = Input((28, 28, 1))

cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn_input_layer)
cnn = MaxPool2D(2)(cnn)

cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = MaxPool2D(2)(cnn)

cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)

cnn = Flatten()(cnn)

cnn = Dense(64, kernel_initializer='he_normal')(cnn)
cnn = BatchNormalization()(cnn)
cnn = ReLU()(cnn)
cnn = Dropout(0.2)(cnn)

cnn_output_layer = Dense(NUM_CLASSES, activation = 'softmax')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

optimizer = Nadam(lr=0.0005)
batch_size=128
```

### CNN

- Best Val Loss: 0.0433
- Best Val Accuracy: 0.9874
- 0.045171141624450684, 0.9853174686431885

```
cnn_input_layer = Input((28, 28, 1))

cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn_input_layer)

cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)

cnn = Flatten()(cnn)

cnn = Dense(64, kernel_initializer='he_normal')(cnn)
cnn = BatchNormalization()(cnn)
cnn = ReLU()(cnn)
cnn = Dropout(0.2)(cnn)

cnn_output_layer = Dense(NUM_CLASSES, activation='softmax')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

optimizer = Nadam(lr=0.0005)
batch_size=128
```

### CNN

- Best Val Loss: 0.0427
- Best Val Accuracy: 0.9875
- 0.038617271929979324, 0.9871031641960144

```
cnn_input_layer = Input((28, 28, 1))

cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn_input_layer)
cnn = MaxPool2D(2)(cnn)

cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = MaxPool2D(2)(cnn)

cnn = Flatten()(cnn)

cnn = Dense(64, kernel_initializer='he_normal')(cnn)
cnn = BatchNormalization()(cnn)
cnn = ReLU()(cnn)
cnn = Dropout(0.2)(cnn)

cnn_output_layer = Dense(NUM_CLASSES, activation='softmax')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

optimizer = Nadam(lr=0.0005)
batch_size=128
```

### CNN

- Best Val Loss: 0.0404
- Best Val Accuracy: 0.9885
- 0.04227079078555107, 0.9871031641960144

```
cnn_input_layer = Input((28, 28, 1))

cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn_input_layer)
cnn = MaxPool2D(2)(cnn)

cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = MaxPool2D(2)(cnn)

cnn = Flatten()(cnn)

cnn = Dense(64, kernel_initializer='he_normal')(cnn)
cnn = BatchNormalization()(cnn)
cnn = ReLU()(cnn)
cnn = Dropout(0.2)(cnn)

cnn_output_layer = Dense(NUM_CLASSES, activation='softmax')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

optimizer = Nadam(lr=0.00005)
batch_size=128
```

### CNN: Dense(128)

```
cnn_input_layer = Input((28, 28, 1))

cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn_input_layer)
cnn = MaxPool2D(2)(cnn)

cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = MaxPool2D(2)(cnn)

#cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)

cnn = Flatten()(cnn)

cnn = Dense(128, kernel_initializer='he_normal')(cnn)
cnn = BatchNormalization()(cnn)
cnn = ReLU()(cnn)
cnn = Dropout(0.2)(cnn)

cnn_output_layer = Dense(NUM_CLASSES, activation='softmax')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

optimizer = Nadam(lr=0.00005)
batch_size=128
```

### CNN:

- Best Val Loss: 0.0398
- Best Val Accuracy: 0.9883
- 0.04042045772075653, 0.9892857074737549


```
cnn_input_layer = Input((28, 28, 1))

cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn_input_layer)
cnn = MaxPool2D(2)(cnn)

cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = MaxPool2D(2)(cnn)

cnn = Flatten()(cnn)

cnn = Dense(64, kernel_initializer='he_normal')(cnn)
cnn = BatchNormalization()(cnn)
cnn = ReLU()(cnn)
cnn = Dropout(0.2)(cnn)

cnn_output_layer = Dense(NUM_CLASSES, activation='softmax')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

optimizer = Nadam(lr=0.00005)
batch_size=128

learning_rate_reducer = ReduceLROnPlateau(
    monitor='val_loss', 
    patience=3, 
    factor=0.5, 
    min_lr=5e-10,
    verbose=1, 
)
```

### CNN: MaxPooling with 2strides

- Best Val Loss: 0.0392
- Best Val Accuracy: 0.9886
- 0.040473438799381256, 0.988095223903656

```
cnn_input_layer = Input((28, 28, 1))

cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn_input_layer)
cnn = MaxPool2D(2, 2)(cnn)

cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = MaxPool2D(2, 2)(cnn)

#cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)

cnn = Flatten()(cnn)

cnn = Dense(64, kernel_initializer='he_normal')(cnn)
cnn = BatchNormalization()(cnn)
cnn = ReLU()(cnn)
cnn = Dropout(0.2)(cnn)

cnn_output_layer = Dense(NUM_CLASSES, activation='softmax')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

optimizer = Nadam(lr=0.00005)
batch_size=128

learning_rate_reducer = ReduceLROnPlateau(
    monitor='val_loss', 
    patience=3, 
    factor=0.5, 
    min_lr=5e-10,
    verbose=1, 
)
```

### CNN: No Dropout

- Best Val Loss: 0.0386
- Best Val Accuracy: 0.9883
- Train Loss: 0.0036
- Train Accuracy: 1.0000
- 0.04121801629662514, 0.9871031641960144

```
cnn_input_layer = Input((28, 28, 1))

cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn_input_layer)
cnn = MaxPool2D(2, 2)(cnn)

cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = MaxPool2D(2, 2)(cnn)

cnn = Flatten()(cnn)

cnn = Dense(64, kernel_initializer='he_normal')(cnn)
cnn = BatchNormalization()(cnn)
cnn = ReLU()(cnn)

cnn_output_layer = Dense(NUM_CLASSES, activation='softmax')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

optimizer = Nadam(lr=0.00005)
batch_size=128

learning_rate_reducer = ReduceLROnPlateau(
    monitor='val_loss', 
    patience=3, 
    factor=0.5, 
    min_lr=5e-10,
    verbose=1, 
)
```

### CNN: BatchNorm after Conv layers

Best Val Loss: 0.0365
Best Val Accuracy: 0.9902
Train Loss: 0.0019
Train Accuracy: 1.0000
0.0390562005341053, 0.9890872836112976

```
cnn_input_layer = Input((28, 28, 1))

cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn_input_layer)
cnn = MaxPool2D(2, 2)(cnn)

cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = MaxPool2D(2, 2)(cnn)

cnn = BatchNormalization()(cnn)
cnn = Flatten()(cnn)

cnn = Dense(64, kernel_initializer='he_normal')(cnn)
cnn = BatchNormalization()(cnn)
cnn = ReLU()(cnn)
#cnn = Dropout(0.2)(cnn)

cnn_output_layer = Dense(NUM_CLASSES, activation='softmax')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

optimizer = Nadam(lr=0.00005)
batch_size=128

learning_rate_reducer = ReduceLROnPlateau(
    monitor='val_loss', 
    patience=3, 
    factor=0.5, 
    min_lr=5e-10,
    verbose=1, 
```

### CNN: ELU activation

Best Val Loss: 0.0452
Best Val Accuracy: 0.9873
Train Loss: 0.0041
Train Accuracy: 1.0000
0.043117161840200424, 0.9857142567634583

```
cnn_input_layer = Input((28, 28, 1))

cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn_input_layer)
cnn = MaxPool2D(2, 2)(cnn)

cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = MaxPool2D(2, 2)(cnn)

cnn = BatchNormalization()(cnn)
cnn = Flatten()(cnn)

cnn = Dense(64, kernel_initializer='he_normal')(cnn)
cnn = BatchNormalization()(cnn)
cnn = ELU()(cnn)
#cnn = Dropout(0.2)(cnn)

cnn_output_layer = Dense(NUM_CLASSES, activation='softmax')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

optimizer = Nadam(lr=0.00005)
batch_size=128

learning_rate_reducer = ReduceLROnPlateau(
    monitor='val_loss', 
    patience=3, 
    factor=0.5, 
    min_lr=5e-10,
    verbose=1, 
```

### CNN: 3x BatchNorms

Best Val Loss: 0.0396
Best Val Accuracy: 0.9879
Train Loss: 0.0085
Train Accuracy: 0.9993
0.04150589182972908, 0.9878968000411987

```
cnn_input_layer = Input((28, 28, 1))

cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn_input_layer)
cnn = BatchNormalization()(cnn)
cnn = MaxPool2D(2, 2)(cnn)

cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = MaxPool2D(2, 2)(cnn)

cnn = Flatten()(cnn)

cnn = Dense(64, kernel_initializer='he_normal')(cnn)
cnn = BatchNormalization()(cnn)
cnn = ReLU()(cnn)
cnn = Dropout(0.2)(cnn)

cnn_output_layer = Dense(NUM_CLASSES, activation='softmax')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

optimizer = Nadam(lr=0.00005)
batch_size=128

learning_rate_reducer = ReduceLROnPlateau(
    monitor='val_loss', 
    patience=3, 
    factor=0.5, 
    min_lr=5e-10,
    verbose=1, 
```

### CNN: Adam(1e-3)

Best Val Loss: 0.0323
Best Val Accuracy: 0.9917
Train Loss: 0.0010
Train Accuracy: 0.9998
0.04256705939769745, 0.9908730387687683

```
cnn_input_layer = Input((28, 28, 1))

cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn_input_layer)
cnn = BatchNormalization()(cnn)
cnn = MaxPool2D(2, 2)(cnn)

cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = MaxPool2D(2, 2)(cnn)

cnn = Flatten()(cnn)

cnn = Dense(64, kernel_initializer='he_normal')(cnn)
cnn = BatchNormalization()(cnn)
cnn = ReLU()(cnn)
cnn = Dropout(0.2)(cnn)

cnn_output_layer = Dense(NUM_CLASSES, activation='softmax')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

optimizer = Adam(lr=1e-3)
batch_size=128

learning_rate_reducer = ReduceLROnPlateau(
    monitor='val_loss', 
    patience=3, 
    factor=0.5, 
    min_lr=5e-10,
    verbose=1, 
```

### CNN: Adam(1e-4)

Best Val Loss: 0.0473
Best Val Accuracy: 0.9883
Train Loss: 0.0270
Train Accuracy: 0.9963
0.03927315026521683, 0.9882936477661133

```
cnn_input_layer = Input((28, 28, 1))

cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn_input_layer)
cnn = BatchNormalization()(cnn)
cnn = MaxPool2D(2, 2)(cnn)

cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = MaxPool2D(2, 2)(cnn)

cnn = Flatten()(cnn)

cnn = Dense(64, kernel_initializer='he_normal')(cnn)
cnn = BatchNormalization()(cnn)
cnn = ReLU()(cnn)
cnn = Dropout(0.2)(cnn)

cnn_output_layer = Dense(NUM_CLASSES, activation='softmax')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

optimizer = Adam(lr=1e-3)
batch_size=128

learning_rate_reducer = ReduceLROnPlateau(
    monitor='val_loss', 
    patience=3, 
    factor=0.5, 
    min_lr=5e-10,
    verbose=1, 
```

### CNN: MaxPooling (No Strides)

Best Val Loss: 0.0335
Best Val Accuracy: 0.9907
Train Loss: 0.0018
Train Accuracy: 0.9998
0.031560249626636505, 0.9916666746139526
Kaggle: 0.99064 (Top 35%)

```
cnn_input_layer = Input((28, 28, 1))

cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn_input_layer)
cnn = BatchNormalization()(cnn)
cnn = MaxPool2D(2)(cnn)

cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = MaxPool2D(2)(cnn)

cnn = Flatten()(cnn)

cnn = Dense(64, kernel_initializer='he_normal')(cnn)
cnn = BatchNormalization()(cnn)
cnn = ReLU()(cnn)
cnn = Dropout(0.2)(cnn)

cnn_output_layer = Dense(NUM_CLASSES, activation='softmax')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

optimizer = Adam(lr=1e-3)
batch_size=128

learning_rate_reducer = ReduceLROnPlateau(
    monitor='val_loss', 
    patience=3, 
    factor=0.5, 
    min_lr=5e-10,
    verbose=1, 
```

### CNN: Dropout(0.3)

Best Val Loss: 0.0333
Best Val Accuracy: 0.9914
Train Loss: 0.0052
Train Accuracy: 0.9988
0.03728742152452469, 0.9892857074737549

```
cnn_input_layer = Input((28, 28, 1))

cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn_input_layer)
cnn = BatchNormalization()(cnn)
cnn = MaxPool2D(2)(cnn)

cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = MaxPool2D(2)(cnn)

cnn = Flatten()(cnn)

cnn = Dense(64, kernel_initializer='he_normal')(cnn)
cnn = BatchNormalization()(cnn)
cnn = ReLU()(cnn)
cnn = Dropout(0.3)(cnn)

cnn_output_layer = Dense(NUM_CLASSES, activation='softmax')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

optimizer = Adam(lr=1e-3)
batch_size=128

learning_rate_reducer = ReduceLROnPlateau(
    monitor='val_loss', 
    patience=3, 
    factor=0.5, 
    min_lr=5e-10,
    verbose=1, 
```

### CNN: 2x Denses

Best Val Loss: 0.0360
Best Val Accuracy: 0.9907
Train Loss: 0.0010
Train Accuracy: 0.9999
0.03959345445036888, 0.9894841313362122

```
cnn_input_layer = Input((28, 28, 1))

cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn_input_layer)
cnn = BatchNormalization()(cnn)
cnn = MaxPool2D(2)(cnn)

cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = MaxPool2D(2)(cnn)

cnn = Flatten()(cnn)

cnn = Dense(128, kernel_initializer='he_normal')(cnn)
cnn = BatchNormalization()(cnn)
cnn = ReLU()(cnn)
cnn = Dropout(0.2)(cnn)

cnn = Dense(64, kernel_initializer='he_normal')(cnn)
cnn = BatchNormalization()(cnn)
cnn = ReLU()(cnn)
cnn = Dropout(0.2)(cnn)

cnn_output_layer = Dense(NUM_CLASSES, activation='softmax')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

optimizer = Adam(lr=1e-3)
batch_size=128

learning_rate_reducer = ReduceLROnPlateau(
    monitor='val_loss', 
    patience=3, 
    factor=0.5, 
    min_lr=5e-10,
    verbose=1, 
```

### CNN: 2x Doubled Conv layers (With error)

Best Val Loss: 0.0321
Best Val Accuracy: 0.9918
Train Loss: 0.0008
Train Accuracy: 1.0000
0.03245595842599869, 0.9914682507514954
Kaggle: 0.99207 (Top 27%)

```
cnn_input_layer = Input((28, 28, 1))

cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn_input_layer)
cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn_input_layer)
cnn = BatchNormalization()(cnn)
cnn = MaxPool2D(2)(cnn)

cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = MaxPool2D(2)(cnn)

cnn = Flatten()(cnn)

cnn = Dense(64, kernel_initializer='he_normal')(cnn)
cnn = BatchNormalization()(cnn)
cnn = ReLU()(cnn)
cnn = Dropout(0.2)(cnn)

cnn_output_layer = Dense(NUM_CLASSES, activation='softmax')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

optimizer = Adam(lr=1e-3)
batch_size=128

learning_rate_reducer = ReduceLROnPlateau(
    monitor='val_loss', 
    patience=3, 
    factor=0.5, 
    min_lr=5e-10,
    verbose=1, 
```

### CNN: 2x Doubled Conv layers

Best Val Loss: 0.0276
Best Val Accuracy: 0.9932
Train Loss: 0.0008
Train Accuracy: 1.0000
0.030262907966971397, 0.9928571581840515
Kaggle: 0.99228

```
cnn_input_layer = Input((28, 28, 1))

cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn_input_layer)
cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = MaxPool2D(2)(cnn)

cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = MaxPool2D(2)(cnn)

cnn = Flatten()(cnn)

cnn = Dense(64, kernel_initializer='he_normal')(cnn)
cnn = BatchNormalization()(cnn)
cnn = ReLU()(cnn)
cnn = Dropout(0.2)(cnn)

cnn_output_layer = Dense(NUM_CLASSES, activation='softmax')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')
optimizer = Adam(lr=1e-3)
batch_size=128

learning_rate_reducer = ReduceLROnPlateau(
    monitor='val_loss', 
    patience=3, 
    factor=0.5, 
    min_lr=5e-10,
    verbose=1, 
```

### CNN: He uniform

Best Val Loss: 0.0355
Best Val Accuracy: 0.9919
Train Loss: 0.0006
Train Accuracy: 0.9999
0.03696111589670181, 0.9908730387687683

```
cnn_input_layer = Input((28, 28, 1))

cnn = Conv2D(32, 3, activation='relu', kernel_initializer='he_uniform', padding='same')(cnn_input_layer)
cnn = Conv2D(32, 3, activation='relu', kernel_initializer='he_uniform', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = MaxPool2D(2)(cnn)

cnn = Conv2D(64, 3, activation='relu', kernel_initializer='he_uniform', padding='same')(cnn)
cnn = Conv2D(64, 3, activation='relu', kernel_initializer='he_uniform', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = MaxPool2D(2)(cnn)

cnn = Flatten()(cnn)

cnn = Dense(64, kernel_initializer='he_uniform')(cnn)
cnn = BatchNormalization()(cnn)
cnn = ReLU()(cnn)
cnn = Dropout(0.2)(cnn)

cnn_output_layer = Dense(NUM_CLASSES, activation='softmax')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')
optimizer = Adam(lr=1e-3)
batch_size=128

learning_rate_reducer = ReduceLROnPlateau(
    monitor='val_loss', 
    patience=3, 
    factor=0.5, 
    min_lr=5e-10,
    verbose=1, 
```

### CNN: He normal

Best Val Loss: 0.0289
Best Val Accuracy: 0.9931
Train Loss: 0.0008
Train Accuracy: 0.9999
0.03536858782172203, 0.9916666746139526

```
cnn = Conv2D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same')(cnn_input_layer)
cnn = Conv2D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = MaxPool2D(2)(cnn)

cnn = Conv2D(64, 3, activation='relu', kernel_initializer='he_normal', padding='same')(cnn)
cnn = Conv2D(64, 3, activation='relu', kernel_initializer='he_normal', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = MaxPool2D(2)(cnn)

cnn = Flatten()(cnn)

cnn = Dense(64, kernel_initializer='he_normal')(cnn)
cnn = BatchNormalization()(cnn)
cnn = ReLU()(cnn)
cnn = Dropout(0.2)(cnn)

cnn_output_layer = Dense(NUM_CLASSES, activation='softmax')(cnn)
optimizer = Adam(lr=1e-3)
batch_size=128

learning_rate_reducer = ReduceLROnPlateau(
    monitor='val_loss', 
    patience=3, 
    factor=0.5, 
    min_lr=5e-10,
    verbose=1, 
```

### CNN: 2 Conv with Dropouts (0.2)

Best Val Loss: 0.0245
Best Val Accuracy: 0.9939
Train Loss: 0.0023
Train Accuracy: 0.9996
0.031296174973249435, 0.9930555820465088

```
cnn_input_layer = Input((28, 28, 1))

cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn_input_layer)
cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = MaxPool2D(2)(cnn)
cnn = Dropout(0.2)(cnn)

cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = MaxPool2D(2)(cnn)
cnn = Dropout(0.2)(cnn)

cnn = Flatten()(cnn)

cnn = Dense(64, kernel_initializer='he_normal')(cnn)
cnn = BatchNormalization()(cnn)
cnn = ReLU()(cnn)
cnn = Dropout(0.2)(cnn)

cnn_output_layer = Dense(NUM_CLASSES, activation='softmax')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

optimizer = Adam(lr=1e-3)
batch_size=128

learning_rate_reducer = ReduceLROnPlateau(
    monitor='val_loss', 
    patience=3, 
    factor=0.5, 
    min_lr=5e-10,
    verbose=1, 
```

### CNN: 2 Conv with Dropouts (0.4)

Best Val Loss: 0.0245
Best Val Accuracy: 0.9942
Train Loss: 0.0143
Train Accuracy: 0.9959
0.02474787086248398, 0.9938492178916931

```
cnn_input_layer = Input((28, 28, 1))

cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn_input_layer)
cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = MaxPool2D(2)(cnn)
cnn = Dropout(0.4)(cnn)

cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = MaxPool2D(2)(cnn)
cnn = Dropout(0.4)(cnn)

cnn = Flatten()(cnn)

cnn = Dense(64, kernel_initializer='he_normal')(cnn)
cnn = BatchNormalization()(cnn)
cnn = ReLU()(cnn)
cnn = Dropout(0.4)(cnn)

cnn_output_layer = Dense(NUM_CLASSES, activation='softmax')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

optimizer = Adam(lr=1e-3)
batch_size=128

learning_rate_reducer = ReduceLROnPlateau(
    monitor='val_loss', 
    patience=3, 
    factor=0.5, 
    min_lr=5e-10,
    verbose=1, 
```

### CNN: 3 Double Conv with Droupouts (0.4) and BatchNorms

Best Val Loss: 0.0240
Best Val Accuracy: 0.9943
Train Loss: 0.0078
Train Accuracy: 0.9975
0.02387743815779686, 0.9946428537368774

```
cnn_input_layer = Input((28, 28, 1))

cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn_input_layer)
cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = MaxPool2D(2)(cnn)
cnn = Dropout(0.4)(cnn)

cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = MaxPool2D(2)(cnn)
cnn = Dropout(0.4)(cnn)

cnn = Conv2D(128, 3, activation='relu', padding='same')(cnn)
cnn = Conv2D(128, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = Dropout(0.4)(cnn)

cnn = Flatten()(cnn)

cnn = Dense(64, kernel_initializer='he_normal')(cnn)
cnn = BatchNormalization()(cnn)
cnn = ReLU()(cnn)
cnn = Dropout(0.4)(cnn)

cnn_output_layer = Dense(NUM_CLASSES, activation='softmax')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')
optimizer = Adam(lr=1e-3)
batch_size=128

learning_rate_reducer = ReduceLROnPlateau(
    monitor='val_loss', 
    patience=3, 
    factor=0.5, 
    min_lr=5e-10,
    verbose=1, 
```

### CNN:

Best Val Loss: 0.0227
Best Val Accuracy: 0.9944
Train Loss: 0.0068
Train Accuracy: 0.9980
0.02724132500588894, 0.9928571581840515

```
cnn_input_layer = Input((28, 28, 1))

cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn_input_layer)
cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = MaxPool2D(2)(cnn)
cnn = Dropout(0.4)(cnn)

cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = MaxPool2D(2)(cnn)
cnn = Dropout(0.4)(cnn)

cnn = Conv2D(128, 3, activation='relu', padding='same')(cnn)
cnn = Conv2D(128, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = Dropout(0.4)(cnn)

cnn = Flatten()(cnn)

cnn = Dense(128, kernel_initializer='he_normal')(cnn)
cnn = BatchNormalization()(cnn)
cnn = ReLU()(cnn)
cnn = Dropout(0.4)(cnn)

cnn_output_layer = Dense(NUM_CLASSES, activation='softmax')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')
optimizer = Adam(lr=1e-3)
batch_size=128

learning_rate_reducer = ReduceLROnPlateau(
    monitor='val_loss', 
    patience=3, 
    factor=0.5, 
    min_lr=5e-10,
    verbose=1, 
```

### CNN: 2x Dense layers

Best Val Loss: 0.0292
Best Val Accuracy: 0.9938
Train Loss: 0.0382
Train Accuracy: 0.9915
0.030066518113017082, 0.9934523701667786

Too much regularization

```
cnn_input_layer = Input((28, 28, 1))

cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn_input_layer)
cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = MaxPool2D(2)(cnn)
cnn = Dropout(0.4)(cnn)

cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = MaxPool2D(2)(cnn)
cnn = Dropout(0.4)(cnn)

cnn = Conv2D(128, 3, activation='relu', padding='same')(cnn)
cnn = Conv2D(128, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = Dropout(0.4)(cnn)

cnn = Flatten()(cnn)

cnn = Dense(64, kernel_initializer='he_normal')(cnn)
cnn = BatchNormalization()(cnn)
cnn = ReLU()(cnn)
cnn = Dropout(0.4)(cnn)

cnn = Dense(32, kernel_initializer='he_normal')(cnn)
cnn = BatchNormalization()(cnn)
cnn = ReLU()(cnn)
cnn = Dropout(0.4)(cnn)

cnn_output_layer = Dense(NUM_CLASSES, activation='softmax')(cnn)
cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

optimizer = Adam(lr=1e-3)
batch_size=128

learning_rate_reducer = ReduceLROnPlateau(
    monitor='val_loss', 
    patience=3, 
    factor=0.5, 
    min_lr=5e-10,
    verbose=1, 
```

### CNN:

Best Val Loss: 0.0255
Best Val Accuracy: 0.9945
Train Loss: 0.0115
Train Accuracy: 0.9971
0.030067507177591324, 0.9920634627342224

```
cnn_input_layer = Input((28, 28, 1))

cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn_input_layer)
cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = MaxPool2D(2)(cnn)
cnn = Dropout(0.4)(cnn)

cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = MaxPool2D(2)(cnn)
cnn = Dropout(0.4)(cnn)

cnn = Conv2D(128, 3, activation='relu', padding='same')(cnn)
cnn = Conv2D(128, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = Dropout(0.4)(cnn)

cnn = Flatten()(cnn)

cnn = Dense(64, kernel_initializer='he_normal')(cnn)
cnn = BatchNormalization()(cnn)
cnn = ReLU()(cnn)
cnn = Dropout(0.2)(cnn)

cnn = Dense(32, kernel_initializer='he_normal')(cnn)
cnn = BatchNormalization()(cnn)
cnn = ReLU()(cnn)
cnn = Dropout(0.2)(cnn)

cnn_output_layer = Dense(NUM_CLASSES, activation='softmax')(cnn)
cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

optimizer = Adam(lr=1e-3)
batch_size=128

learning_rate_reducer = ReduceLROnPlateau(
    monitor='val_loss', 
    patience=3, 
    factor=0.5, 
    min_lr=5e-10,
    verbose=1, 
```