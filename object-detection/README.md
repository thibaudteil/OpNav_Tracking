## Testing TensorFlow Object Detection API with craters

Based on this [tutorial](https://medium.com/towards-data-science/building-a-toy-detector-with-tensorflow-object-detection-api-63c0fdf2ac95).

* `create_pet_tf_record.py` - Licensed under Apache. See file for more details
This contains code to convert image and annotations into a TFRecord file as required by the API.

## Installation instructions
You need to setup your environment before you can start using this.
Detailed instructions can be found [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)

## Other dependencies
You need to download a pre-trained model from [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). Choose the model corresponding to the `*.config` file.
Extract the contents (it should contain `.ckpt` files) into `./.ckpt/` directory.

## Running

Assuming you have already installed the necessary dependencies and setup the `.config` file to reflect changes in your environment (such as path to `.record` file, number of classes, etc.).

```bash
$ python train.py --logtostderr --train_dir=.ckpt-trained/ --pipeline_config_path=<your_config_file>
```
`train_dir` is the path where you want to store *your* training checkpoints.
