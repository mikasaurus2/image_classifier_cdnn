from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# load and prepare the image
def load_image(filename):
   img = load_img(filename, target_size=(200, 200))
   img = img_to_array(img)
   # reshape into a single sample with 3 channels
   img = img.reshape(1, 200, 200, 3)
   # center pixel data
   img = img.astype('float32')
   img = img - [123.68, 116.779, 103.939]
   return img

# load an image and predict the class
def run_example():
   img = load_image('dog_sample_image.jpg')
   model = load_model('initial_model.keras')
   result = model.predict(img)
   # Based on the way the training directories are set up,
   # cat corresponds to 0
   # dog corresponds to 1
   print(result[0])

run_example()
