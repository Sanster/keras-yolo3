"""
Retrain the YOLO model for your own dataset.
"""
import os

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras import optimizers

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data

TRAIN_EPOCHS = 600
TRAINABLE_LAYER = 20
LEARNING_RATE = 0.0001


def _main():
    annotation_path = 'train.txt'
    log_dir = 'logs/000/'
    classes_path = 'model_data/coco_classes_plus.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)

    input_shape = (416, 416)  # multiple of 32, hw

    is_tiny_version = len(anchors) == 6  # default setting
    create_func = create_tiny_model if is_tiny_version else create_model
    model = create_func(input_shape, anchors, len(class_names),
                        load_pretrained=True, freeze_body=True)
    model.summary()
    train(model, annotation_path, input_shape, anchors, len(class_names), log_dir=log_dir)


def train(model, annotation_path, input_shape, anchors, num_classes, log_dir='logs/'):
    '''retrain/fine-tune the model'''
    adam = optimizers.Adam(lr=LEARNING_RATE)
    # use custom yolo_loss Lambda layer.
    model.compile(optimizer=adam, loss={'yolo_loss': lambda y_true, y_pred: y_pred})

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=30)

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1, mode='auto')

    batch_size = 32
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.shuffle(lines)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    model.fit_generator(data_generator_wrap(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=data_generator_wrap(lines[num_train:], batch_size, input_shape, anchors,
                                                            num_classes),
                        validation_steps=max(1, num_val // batch_size),
                        epochs=TRAIN_EPOCHS,
                        initial_epoch=0,
                        callbacks=[logging, checkpoint])
    model.save_weights('./model_data/trained_weights_20_layer.h5')
    # Further training.


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=True):
    '''create the training model'''
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], \
                           num_anchors // 3, num_classes + 5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors // 3, num_classes)

    if load_pretrained:
        weights_path = os.path.join('model_data', 'yolo_weights.h5')
        if not os.path.exists(weights_path):
            print("CREATING WEIGHTS FILE" + weights_path)
            yolo_path = os.path.join('model_data', 'yolo.h5')
            orig_model = load_model(yolo_path, compile=False)
            orig_model.save_weights(weights_path)
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        if freeze_body:
            # Do not freeze 3 output layers.
            print("Total layer: %d" % len(model_body.layers))
            for i in range(len(model_body.layers) - TRAINABLE_LAYER):
                model_body.layers[i].trainable = False

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=True):
    '''create the training model, for Tiny YOLOv3'''
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h // {0: 32, 1: 16}[l], w // {0: 32, 1: 16}[l], \
                           num_anchors // 2, num_classes + 5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors // 2, num_classes)

    if load_pretrained:
        weights_path = os.path.join('model_data/', 'tiny_yolo_weights.h5')
        if not os.path.exists(weights_path):
            print("CREATING WEIGHTS FILE" + weights_path)
            yolo_path = os.path.join('model_data', 'tiny_yolo.h5')
            orig_model = load_model(yolo_path, compile=False)
            orig_model.save_weights(weights_path)
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        if freeze_body:
            # Do not freeze 2 output layers.
            for i in range(len(model_body.layers) - 2):
                model_body.layers[i].trainable = False

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    np.random.shuffle(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            i %= n
            image, box = get_random_data(annotation_lines[i], input_shape)
            image_data.append(image)
            box_data.append(box)
            i += 1
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def data_generator_wrap(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


if __name__ == '__main__':
    _main()
