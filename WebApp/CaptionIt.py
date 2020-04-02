import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences

import warnings
warnings.filterwarnings('ignore')

## Loading word to index mapping
with open('./resources/word2idx.pkl','rb') as f:
    word2idx=pickle.load(f)

## Loading index to word mapping
with open('./resources/idx2word.pkl','rb') as f:
    idx2word=pickle.load(f)


maxlen = 84
model = load_model('./models_weights/model_6.h5')
#model._make_predict_function()
featureExtract_model = load_model("./resources/resNet50.h5")
#featureExtract_model._make_predict_function()

def preprocess_image(img_path):
    img = image.load_img(img_path,target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    img = preprocess_input(img)  ## preprocessing for ResNet50
    return img
def encode_image(img_path):
    img = preprocess_image(img_path)
    img_feature_vector = featureExtract_model.predict(img)
    img_feature_vector = img_feature_vector.reshape((-1,))
    return img_feature_vector

def predict_caption(image_path):
    
    photo = encode_image(image_path).reshape(1,2048)
    
    input_text = '<s>'
    for _ in range(maxlen):
        sequence = [word2idx[word] for word in input_text.split() if word in word2idx]
        sequence = pad_sequences([sequence],maxlen=maxlen,padding='post')
        
        ypred = model.predict([photo,sequence])
        ypred = ypred.argmax()
        word = idx2word[ypred]
        
        input_text +=' ' + word
        
        if word == '<e>':
            break
            
    final_caption=input_text.split()[1:-1]
    final_caption=' '.join(final_caption)
    
    return final_caption.capitalize()
    # img=plt.imread(image_path)
    # plt.imshow(img)
    # plt.show()
