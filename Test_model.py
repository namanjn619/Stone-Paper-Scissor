from keras.models import load_model
from keras.preprocessing import image
import numpy as np

REV_CLASS_MAP = {
    0: "Rock",
    1: "Paper",
    2: "Scissor",
}


def mapper(val):
    return REV_CLASS_MAP[val]


model = load_model("C://Users//a//Desktop//VScode//Rock Paper Scissor//My Game//rock-paper-scissors-model.h5")
test_image = image.load_img("C://Users//a//Desktop//VScode//Rock Paper Scissor//My Game//Test Images//Rock.png", target_size=(227,227))

test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)

result = model.predict(test_image)

move_code = np.argmax(result[0])
move = mapper(move_code)

print("Predicted: {}".format(move))

