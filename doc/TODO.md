- normalize the audio [-1.0, 1.0] (aparently load_wav is doing this)


# TensorFlow
- record the mean and standard deviation of the audio data at training time, and include in the model csv
  - normalize data coming from the mic (subtract the mean)
- Maybe run a high pass filter, say, 10khz, through ALL the audio, then train on that (also need to during inference) 
[x] - verify the output of the model using some known files
[x] - build something that can listen in realtime and classify once every 1 second
[x] - normalize, and resample to 16Khz single channel
- try TFLite (AudioClassifier), any good? useful?
  - https://www.tensorflow.org/lite/inference_with_metadata/task_library/audio_classifier
[x] - build/print/plot a confusion matrix
  [x] - https://www.tensorflow.org/tutorials/audio/simple_audio#display_a_confusion_matrix