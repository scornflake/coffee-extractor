- normalize the audio [-1.0, 1.0]


# TensorFlow
- verify the output of the model using some known files
  - perhaps as part of training
  - so I can understand the results from inference
  - itd be nice to print the names of the labels as well
- build something that can listen in realtime and classify once every 1 second
- normalize, and resample to 16Khz single channel
- try TFLite (AudioClassifier), any good? useful?
  - https://www.tensorflow.org/lite/inference_with_metadata/task_library/audio_classifier
- build/print/plot a confusiono matrix
  - https://www.tensorflow.org/tutorials/audio/simple_audio#display_a_confusion_matrix