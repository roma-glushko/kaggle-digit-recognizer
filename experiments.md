# Experiment Notes

## SVM

### PCA(n_comp=154), RBF SVC - 0.9925, 0.9762, 0.97610 [training set]
### PCA(n_comp=154), RBF SVC - 0.9927, 0.9712 [reduced traning set]
### PCA(n_comp=154), SVC(RBF, gamma=0.05) - 0.9991, 0.9736 [reduced traning set]
### PCA(n_comp=154), SVC(RBF, gamma=0.1) - ?, 0.92?? [reduced traning set]
### PCA(n_comp=154), SVC(RBF, gamma=0.05, C=0.001) - 0.1116, 0.1116 [reduced traning set]
### PCA(n_comp=154), SVC(RBF, gamma=0.05, C=0.01) -  0.3920, 0.3179 [reduced traning set]
### PCA(n_comp=154), SVC(RBF, gamma=0.05, C=5) - 1, 0.9738 [reduced traning set]

## MLP

### Denses(200, relu) + Dropout(.2) + BatchNorm + Dense(150, relu) + Dropout(.2) + BatchNorm

Best Validation Loss: 0.0810
Best Validation Accuracy: 0.9771
Test Validation Loss: 0.0957236960530281
Test Accuracy: 0.9724206328392029

### Denses(200) + BatchNorm + LeakyReLu + Dropout(.2) + Dense(150) + BatchNorm + LeakyReLu + Dropout(.2) 

Best Validation Loss: 0.1024
Best Validation Accuracy: 0.9723
Test Validation Loss: 0.11114469170570374
Test Accuracy: 0.9676587581634521

### Denses(200) + BatchNorm + ReLu + Dropout(.2) + Dense(150) + BatchNorm + ReLu + Dropout(.2) 
Best Validation Loss: 0.0800
Best Validation Accuracy: 0.9789
Test Validation Loss: 0.09856043756008148
Test Accuracy: 0.9724206328392029

### Denses(200) + BatchNorm + ReLu + Dropout(.2) + Dense(200) + BatchNorm + ReLu + Dropout(.2) 
Best Validation Loss: 0.0807
Best Validation Accuracy: 0.9787
Test Validation Loss: 0.08998408168554306
Test Accuracy: 0.9740079641342163

### Denses(512) + BatchNorm + ReLu + Dropout(.2)
Best Validation Loss: 0.0902
Best Validation Accuracy: 0.9773
Test Validation Loss: 0.10907424241304398
Test Accuracy: 0.9708333611488342

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

### CNN: Data Augmentation

Best Val Loss: 0.0221
Best Val Accuracy: 0.9956
Train Loss: 0.0250
Train Accuracy: 0.9943
0.026448724791407585, 0.9930555820465088

```
cnn_input_layer = Input((28, 28, 1))

cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn_input_layer)
cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = MaxPool2D(2)(cnn)
cnn = Dropout(0.35)(cnn)

cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = MaxPool2D(2)(cnn)
cnn = Dropout(0.35)(cnn)

cnn = Conv2D(128, 3, activation='relu', padding='same')(cnn)
cnn = Conv2D(128, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = Dropout(0.35)(cnn)

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
data_augmentator = ImageDataGenerator(
    rotation_range = 10,  
    zoom_range = 0.1, 
    width_shift_range = 0.1, 
    height_shift_range = 0.1
)

```

### CNN

Best Val Loss: 0.0216
Best Val Accuracy: 0.9957
Train Loss: 0.0229
Train Accuracy: 0.9943
0.026916267350316048, 0.9930555820465088

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

cnn = Conv2D(128, 3, activation='relu', padding='same')(cnn)
cnn = Conv2D(128, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = Dropout(0.2)(cnn)

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
data_augmentator = ImageDataGenerator(
    rotation_range = 10,  
    zoom_range = 0.1, 
    width_shift_range = 0.1, 
    height_shift_range = 0.1
)
```

### CNN: MaxPool to 3rd Conv layer

Best Val Loss: 0.0221
Best Val Accuracy: 0.9951
Train Loss: 0.0162
Train Accuracy: 0.9958
0.024651357904076576, 0.9944444298744202

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

cnn = Conv2D(128, 3, activation='relu', padding='same')(cnn)
cnn = Conv2D(128, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = MaxPool2D(2)(cnn)
cnn = Dropout(0.2)(cnn)

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
data_augmentator = ImageDataGenerator(
    rotation_range = 10,  
    zoom_range = 0.1, 
    width_shift_range = 0.1, 
    height_shift_range = 0.1
)
```

### CNN: GlobalAvgPool

Best Val Loss: 0.0211
Best Val Accuracy: 0.9948
Train Loss: 0.0095
Train Accuracy: 0.9972
0.02560603804886341, 0.9928571581840515

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

cnn = Conv2D(128, 3, activation='relu', padding='same')(cnn)
cnn = Conv2D(128, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = GlobalMaxPooling2D()(cnn)

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
data_augmentator = ImageDataGenerator(
    rotation_range = 10,  
    zoom_range = 0.1, 
    width_shift_range = 0.1, 
    height_shift_range = 0.1
)
```

### CNN

Best Val Loss: 0.0220
Best Val Accuracy: 0.9946
Train Loss: 0.0154
Train Accuracy: 0.9957
0.024416828528046608, 0.9936507940292358

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

cnn = Conv2D(128, 3, activation='relu', padding='same')(cnn)
cnn = Conv2D(128, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = GlobalAveragePooling2D()(cnn)

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
data_augmentator = ImageDataGenerator(
    rotation_range = 10,  
    zoom_range = 0.1, 
    width_shift_range = 0.1, 
    height_shift_range = 0.1
)
```

### CNN

Best Val Loss: 0.0199
Best Val Accuracy: 0.9949
Train Loss: 0.0158
Train Accuracy: 0.9949
0.023942016065120697, 0.9946428537368774

```
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D

cnn_input_layer = Input((28, 28, 1))

cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn_input_layer)
cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = MaxPool2D(2)(cnn)
cnn = Dropout(0.25)(cnn)

cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = MaxPool2D(2)(cnn)
cnn = Dropout(0.25)(cnn)

cnn = Conv2D(128, 3, activation='relu', padding='same')(cnn)
cnn = Conv2D(128, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = MaxPool2D(2)(cnn)
cnn = Dropout(0.25)(cnn)

cnn = Flatten()(cnn)

cnn = Dense(128, kernel_initializer='he_normal')(cnn)
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
data_augmentator = ImageDataGenerator(
    rotation_range = 10,  
    zoom_range = 0.1, 
    width_shift_range = 0.1, 
    height_shift_range = 0.1
)
```

### CNN: BatchNorm after all Conv layers

Best Val Loss: 0.0194
Best Val Accuracy: 0.9950
Train Loss: 0.0177
Train Accuracy: 0.9943
0.020568035542964935, 0.9946428537368774

```
cnn_input_layer = Input((28, 28, 1))

# layer 1
cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn_input_layer)
cnn = BatchNormalization()(cnn)

cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)

cnn = MaxPool2D(2)(cnn)
cnn = Dropout(0.25)(cnn)

# layer 2
cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)

cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)

cnn = MaxPool2D(2)(cnn)
cnn = Dropout(0.25)(cnn)

# layer 3
cnn = Conv2D(128, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)

cnn = Conv2D(128, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)

cnn = MaxPool2D(2)(cnn)
cnn = Dropout(0.25)(cnn)

cnn = Flatten()(cnn)

# layer 4
cnn = Dense(128, kernel_initializer='he_normal')(cnn)
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
data_augmentator = ImageDataGenerator(
    rotation_range = 10,  
    zoom_range = 0.1, 
    width_shift_range = 0.1, 
    height_shift_range = 0.1
)
```

### CNN: Exponential Learning Rate Scheduler

Best Val Loss: 0.0228
Best Val Accuracy: 0.9954
Train Loss: 0.0181
Train Accuracy: 0.9943
0.02234075963497162, 0.9942460060119629

```
cnn_input_layer = Input((28, 28, 1))

# layer 1
cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn_input_layer)
cnn = BatchNormalization()(cnn)

cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)

cnn = MaxPool2D(2)(cnn)
cnn = Dropout(0.25)(cnn)

# layer 2
cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)

cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)

cnn = MaxPool2D(2)(cnn)
cnn = Dropout(0.25)(cnn)

# layer 3
cnn = Conv2D(128, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)

cnn = Conv2D(128, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)

cnn = MaxPool2D(2)(cnn)
cnn = Dropout(0.25)(cnn)

cnn = Flatten()(cnn)

# layer 4
cnn = Dense(128, kernel_initializer='he_normal')(cnn)
cnn = BatchNormalization()(cnn)
cnn = ReLU()(cnn)
cnn = Dropout(0.3)(cnn)

cnn_output_layer = Dense(NUM_CLASSES, activation='softmax')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

optimizer = Adam(lr=1e-3)

learning_rate_scheduler = LearningRateScheduler(lambda epoch: 1e-3 * 0.95 ** epoch)

batch_size=128
data_augmentator = ImageDataGenerator(
    rotation_range = 10,  
    zoom_range = 0.1, 
    width_shift_range = 0.1, 
    height_shift_range = 0.1
)
```

### CNN: Replaced MaxPools with X-C5S2ReLU

Best Val Loss: 0.0223
Best Val Accuracy: 0.9951
Train Loss: 0.0075
Train Accuracy: 0.9976
0.020472226664423943, 0.9952380657196045

```
cnn_input_layer = Input((28, 28, 1))

# layer 1
cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn_input_layer)
cnn = BatchNormalization()(cnn)

cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)

#cnn = MaxPool2D(2)(cnn)
cnn = Conv2D(32, 5, strides=2, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = Dropout(0.25)(cnn)

# layer 2
cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)

cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)

cnn = Conv2D(64, 5, strides=2, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = Dropout(0.25)(cnn)

# layer 3
cnn = Conv2D(128, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)

cnn = Conv2D(128, 3, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)

cnn = Conv2D(128, 5, strides=2, activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = Dropout(0.25)(cnn)

cnn = Flatten()(cnn)

# layer 4
cnn = Dense(128, kernel_initializer='he_normal')(cnn)
cnn = BatchNormalization()(cnn)
cnn = ReLU()(cnn)
cnn = Dropout(0.3)(cnn)

cnn_output_layer = Dense(NUM_CLASSES, activation='softmax')(cnn)

cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')

optimizer = Adam(lr=1e-3)

learning_rate_scheduler = LearningRateScheduler(lambda epoch: 1e-3 * 0.95 ** epoch)

batch_size=128
data_augmentator = ImageDataGenerator(
    rotation_range = 10,  
    zoom_range = 0.1, 
    width_shift_range = 0.1, 
    height_shift_range = 0.1
)
```

### CNN: Ensamble of 10 CNNs

             Train  Validation 

      loss    0.0228    0.0137
  accuracy    0.9933    0.9969

Kaggle: 0.99657 (Top 7%)

```
cnn_input_layer = Input((28, 28, 1))

    # layer 1
    cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn_input_layer)
    cnn = BatchNormalization()(cnn)

    cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn)
    cnn = BatchNormalization()(cnn)

    #cnn = MaxPool2D(2)(cnn)
    cnn = Conv2D(32, 5, strides=2, activation='relu', padding='same')(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = Dropout(0.4)(cnn)

    # layer 2
    cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
    cnn = BatchNormalization()(cnn)

    cnn = Conv2D(64, 3, activation='relu', padding='same')(cnn)
    cnn = BatchNormalization()(cnn)

    #cnn = MaxPool2D(2)(cnn)
    cnn = Conv2D(64, 5, strides=2, activation='relu', padding='same')(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = Dropout(0.4)(cnn)

    # layer 3
    cnn = Conv2D(128, 3, activation='relu', padding='same')(cnn)
    cnn = BatchNormalization()(cnn)

    cnn = Conv2D(128, 3, activation='relu', padding='same')(cnn)
    cnn = BatchNormalization()(cnn)

    #cnn = MaxPool2D(2)(cnn)
    cnn = Conv2D(128, 5, strides=2, activation='relu', padding='same')(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = Dropout(0.4)(cnn)

    cnn = Flatten()(cnn)

    # layer 4
    cnn = Dense(128, kernel_initializer='he_normal')(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = ReLU()(cnn)
    cnn = Dropout(0.4)(cnn)

    cnn_output_layer = Dense(NUM_CLASSES, activation='softmax')(cnn)
    
    cnn_model = Model(cnn_input_layer, cnn_output_layer, name='CNN')
    cnn_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-3), metrics=['accuracy'])
```

```
learning_rate_scheduler = LearningRateScheduler(lambda epoch: 1e-3 * 0.95 ** epoch)
batch_size = 128
data_augmentator = ImageDataGenerator(
    rotation_range = 10,  
    zoom_range = 0.1, 
    width_shift_range = 0.1, 
    height_shift_range = 0.1
)
```