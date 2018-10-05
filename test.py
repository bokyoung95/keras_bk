from quiver_engine.server import launch
from keras.models import model_from_json

json_file = open("model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model_weights.h5")

#model = load_model('VGG16_emotion.h5')
launch(model, temp_folder='./tmp', input_folder='./data_bk/1neutral', port=2002)
