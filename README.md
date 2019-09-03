Documentation to be updated. 

# Wake Word Detection


Detect the word "activate" from audio file or audio stream with deep RNN.

---

## Demo/Result

<p align="center">
  <a href="https://youtu.be/GQgyfuL00YA"><b>Watch the Result (not cherry picked)</b></a>
  <br>
  <a href="https://youtu.be/GQgyfuL00YA"><img width="50%" height="50%" src="assets/result.png" title="wake word detection" alt="Video Missing"></a>
</p>

---

## Model

<p align="center"><img src="assets/model.png" width="75%" height="75%"></p>

---

## Try it Yourself

Requirements in [`info/requirements.txt`](https://github.com/Jacklu0831/Wake-Word-Detection/blob/master/info/requirements.txt).

### Make Your Custom Data

### Train

### Test

---

## Files

<pre>
README.md                  - self
measure_surrounding.py     - script for logging and averaging the volume of surrounding
preprocess_data.ipynb      - colab notebook for playing with audio files and producing data (audio synthesis)
wake_word_detection.ipynb  - colab notebook for building model, training model and comparing model performances
real-time-detection.py     - script for trying out the model through terminal (audio stream)

assets                     - images and videos for this markdown
info                       - contains the model summary and requirements for running this program
input                      - contains input sample data and my own data
output                     - the mp4 file of my video demo
model                      - contains the trained general model and further trained model for myself
</pre>

## Resources

- [pydub](https://github.com/jiaaro/pydub)
- [CS230 slides](http://cs230.stanford.edu/fall2018/slides_week2.pdf)
- [Blog on trigger word recognition](https://www.dlology.com/blog/how-to-do-real-time-trigger-word-detection-with-keras/)
- [Coursera sequence models specialization](https://www.coursera.org/learn/nlp-sequence-models)
