from tensorflow.keras.applications import DenseNet121,DenseNet201
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import  Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

img_size = 299

def get_model(model_name) :
    if model_name == 'MobileNet' :
        base_model=MobileNet(weights='imagenet',include_top=False,input_shape=(img_size,img_size,3))
    elif model_name == 'VGG16' :
        base_model=vgg16.VGG16(weights='imagenet',include_top=False,input_shape=(img_size,img_size,3))
    elif model_name == 'DenseNet' :
        base_model = DenseNet121(weights='imagenet',include_top=False,input_shape=(img_size,img_size,3))
    elif model_name == 'DenseNet201' :
        base_model = DenseNet201(weights='imagenet',include_top=False,input_shape=(img_size,img_size,3))
    elif model_name == 'Inception' :
        base_model=InceptionV3(weights='imagenet',include_top=False,input_shape=(img_size,img_size,3))
    elif model_name == 'ResNet' :
        base_model = ResNet50(weights='imagenet',include_top=False,input_shape=(img_size,img_size,3))
    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(512,activation='relu')(x)
    x=Dropout(0.3)(x)
    x=Dense(256,activation='relu')(x) #dense layer 2
    preds=Dense(4,activation='softmax')(x)
    model=Model(inputs=base_model.input,outputs=preds,name=model_name)
    model.compile(optimizer='Adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
    return model
