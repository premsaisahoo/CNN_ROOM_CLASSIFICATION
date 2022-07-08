#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: premsai
"""

import numpy as np
from keras.models import load_model
from keras.preprocessing import image

class room:
    def __init__(self,filename):
        self.filename =filename


    def predictionroom(self):
        # load model
        model = load_model('mymodel.h5')

        # summarize model
        #model.summary()
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (60, 60))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)

        if result[0][0] == 0:
            prediction = 'bed_room'
            return [{ "image" : prediction}]

        elif result[0][0] == 1:
            prediction = 'dining_room'
            return [{ "image" : prediction}]
        else:
            prediction = 'kitchen_room'
            return [{ "image" : prediction}]


