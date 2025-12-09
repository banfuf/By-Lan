import tensorflow as tf 
import keras
mnist = keras.datasets.mnist
(train_image,train_labals),(test_image,test_labals) = mnist.load_data()
train_image = train_image / 255
test_image = test_image / 255 
model =  keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),
                           keras.layers.Dense(128,activation='relu'),
                           keras.layers.Dense(10,activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_image,train_labals,epochs=5)
test_loss,test_acc = model.evaluate(test_image,test_labals,verbose=2)
model.save('model.h5')
print("模型已保存到model.h5")