
  

# 3rd Year Project
  

## Research
### Datasets
[Lip Reading Datasets](https://www.bbc.co.uk/rd/projects/lip-reading-datasets)
[LRS2](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html)
[LRW](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html)
### Similar Work
https://khazit.github.io/Lip2Word/

https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cit2.12131

https://paperswithcode.com/task/lipreading

https://medium.com/@allenye66/lip-reading-using-computer-vision-and-deep-learning-e602f1bebee1

https://www.mdpi.com/1418782

https://arxiv.org/pdf/1703.04105v4.pdf

### Automatic Speech Recognition using CTC

Link to article [here](https://keras.io/examples/audio/ctc_asr/)

**Summary**:

- Article instead was surrounding audio speech recognition
- Article used CTC for a loss function
	- *CTC is an algorithm used to train deep neural networks in speech recognition, handwriting recognition and other sequence problems. CTC is used when we don’t know how the input aligns with the outpu*
	- This would therefore suggest that CTC would be largely beneficial for my project
	- We know how individual words align with lip shapes but what about breaking those words down? We could use CTC for training Phoneme/Viseme based lip reading instead
- The article trained on letters individually
	- Vocabulary used is a-z, 1-9, etc
	- With CTC this is to find the unknown alignment between words and letters
	- This is shown to not be perfect, sometimes struggling with homophones such as "cer" and "sir" in the final few examples
	- Training on whole words or possibly visemes would therefore reach a similar problem
	- Likely a world model (to find which word is more likely to appear in human language) would help to overcome this

### Viseme Definition

- Article [here](https://repository.uel.ac.uk/download/6981fc25223b614bcb5751bd44143cfcda91484bd32bc4f1d7816c09e5f60a66/350438/Which%20phoneme-to-viseme.pdf) suggests that the best mapping of Phonemes to Visemes is that of *Lee, S., Yook, D.: Audio-to-visual conversion using hidden markov models. In:PRICAI 2002: Trends in Artificial Intelligence. Springer (2002) 563–570*
- This gives a list of 6 consonant, 7 vowel and 1 silence visemes
- Therefore for viseme based learning this definition of visemes would be best to use
- This does give rise to a new problem: converting from visemes to phonemes and/or words: an NLP problem

### English Words to Phonemes

- To be able to do viseme based learning we must convert from written English words to Phonemes
	- This is for automatic data generation where we have the word being spoken and must convert to a viseme
	- Pipeline: Written word -> Phoneme components -> Viseme components
	- We can then classify that particular frame as the specific viseme and train with this
- There exists a Python library called [eng-to-ipa](https://pypi.org/project/eng-to-ipa/) which has the purpose of doing this conversion
- Examples/tutorial can be found [here](https://www.geeksforgeeks.org/convert-english-text-into-the-phonetics-using-python/)

### Lipreading Using Temporal Convolutional Networks

- [Link](https://ieeexplore.ieee.org/abstract/document/9053841)
- Using TCNs may be better than LSTMs

### Background
#### Visemes
[What are Visemes?](https://en.wikipedia.org/wiki/Viseme)
- Visemes and phonemes don’t line up perfectly
- Some sounds may vary compared with their phoneme
- Some things said look the same to lip readers
- Could require within the project some kind of bias between words that are more likely to occur in the real world: may need a world model or language model
### Technologies
#### [MediaPipe](https://github.com/google/mediapipe/blob/master/docs/solutions/face_mesh.md)
- Tool used for detecting human keypoints
- Has capability to create face meshes and detect human face keypoints
- Guide for the face [landmark numbers](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)
#### [DLIB](http://dlib.net/)

- [Specific face recognition example](http://dlib.net/face_recognition.py.html)
- [Guide on localising to faces](https://pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/)
- DLIB is a tool used for detecting human faces
- Very similar to MediaPipe, could be used in the same way
#### [PyTorch](https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html)

- Can be used to create basic models and neural networks
#### [Collab](https://colab.google/)

- Cells for executing code

- Can pay for using GPUs for training models more easily
	- Already get a free GPU to begin with
	- Further computing power can be subscribed to [here](https://colab.research.google.com/signup/pricing?utm_source=notebook_settings&utm_medium=link&utm_campaign=premium_gpu_selector)

- Can store data online to use
## Development Plan
### Initial Plan

**Aims**

- Detect and localise to a person’s lips and record lip movement across frames
- Model should be made that is able to interpret lip movement as phonemes or words
- Need some form of natural language processing in order to put together phonemes or words into valid sentences to display
- System should have a User Interface to display the current video feed as well as the words being spoken
**Steps**

- Use Mediapipe to localise and detect user lips and convert lip movement to valid data of some form
- Train a model to use the lips data to recognise and interpret lip movement as either phonemes or words
- String together words and phonemes to create valid words and sentences
- Create a user interface to display video feed and words being interpreted
### Plan Used
**Steps**
- Split the data into two sets: training & testing
	- Training: For creating the models (will be used for train/test/val split for creating the models)
	- Testing: For evaluating the trained models against completely unseen data
- Use MediaPipe & DLIB to generate two extra sets of data: keypoint data centred around the lips
- Normalise keypoint data
- Train two separate neural networks using the keypoint data
- Compare basic metrics of the models
- Create visualisations of the feature vectors generated by the model to better understand the models and model performance
- Test the two models against completely unseen data to better compare their performance
**Extensional Steps**
- Create a live feed/better UI to better represent the model running in real time
- Try creating a model instead by training with phonemes/visemes
- Try creating a model instead by training on lip images, rather than lip keypoints

## Start of Development
### Technology Decisions
#### MediaPipe vs DLIB
- MediaPipe & DLIB are used for the same thing: face recognition and representing the face using features
- Which to use?
	- MediaPipe has 40 keypoints just for lips (including the inner and outer lip areas)
	- DLIB has only 19 keypoints for the lips
**Conclusion**:
- Mediapipe offers more keypoints and therefore more data to train models with
	- This will result typically in better performing models
- DLIB could be useful to train with also to compare the metrics of the two models and see which is better
- May be that models provided by DLIB are better than those for MediaPipe
#### Visualisation Methods
- To visualize we want to be able to plot data in 2D space based on feature vectors generated by trained models
- Therefore we need a dimensionality reduction algorithm
- Algorithms found: TSNE or UMAP
	- [Research](https://blog.bioturing.com/2022/01/14/umap-vs-t-sne-single-cell-rna-seq-data-visualization/)
**UMAP or TSNE**:
- UMAP is faster & computationally cheaper
	- This is more obvious on very large datasets: like the one being used for this project
- UMAP can give a better presentation of the global structure of data
	- UMAP is easier to read and more reliable to draw conclusions from than TSNE
**Conclusion**:
- UMAP is faster and more reliable and therefore the better method
- Both methods should be used for visualization to compare the results and get a better idea about the layout of feature vectors
- It should be taken into account that conclusions reached from UMAP may be more reliable
### Methodology
#### Full Words, Phonemes or Visemes
Could train on full words, phonemes (or visemes):
- Full words would be video data rather than single images and single images
- Full words would be more static, built to learn just the words input into the model and no more. It couldn’t extrapolate at all
- Alternatively could train on phonemes, or their visual counterparts of visemes
- Phonemes would instead be training against single images, one at a time
- Phonemes would required adjusting the dataset, splitting up videos possible manually into different phonemes being said
	- This would take a lot of time
	- Possibly not accurate
- Phonemes would be more dynamic. There are only 44 phonemes in English compared to 171,476 words in the English dictionary therefore it would be better to train on phonemes
- Phonemes then also poses the challenge of putting the phonemes together into an understandable sentence
	- Hard than with words possibly
	- Would require a comprehensive list of the exact phonemes making up each word. Dictionaries provide this but not necessarily for every word
	- There are some online tools for this like [here](https://tophonetics.com/)
- Phonemes are spoken noises instead training could be carried out on visemes although, the number of visemes and what they are varies
	- [Visemes information](https://en.wikipedia.org/wiki/Viseme)
**Conclusion**:
- To begin with train on full words
	- Need to see accuracy of this, maybe it will be good enough
	- Would require less effort on the data to begin with, especially as most data is instead labelled with full words rather than phonemes
	- Easier to test manually as it takes more time to figure out the exact words from phonemes rather than just the words being spoken
- May later try to train on phonemes
- 	 require significantly more work, especially for data generation
- It would be unwise to train on visemes
	- Could look into trying to replicate some other system for visemes (a defined set of visemes, eg the [Classic Disney viseme](https://docs.cryengine.com/display/SDKDOC2/Phonemes+and+Visemes#:~:text=The%20m%20in%20mom%20and,usually%20around%2010%2D14%20visemes.))
- Training on either phonemes or visemes would be very difficult
	- Very difficult to tell the difference between individual parts of words (Eg, [p, b, m])
	- Easier to look at the word as a whole
#### Training on Images or Keypoints
**Two methods**:
- Could use tools like MediaPipe/DLIB to identify lips within the data, crop these as images and train off of the image data
- Could use tools like MediaPipe/DLIB to identify lips within the data, save feature vectors of the positions of lip keypoints with reference to some anchor (Eg, the nose, the face as a whole, to each other), train off of feature vectors
**Evaluation**:
- Images would need to be augmented before being passed into a neural network
	- Data passed into an NN must be of the same size, therefore as lip crops will be different sizes we will need to augment them
	- Possible methods: Cropping, scaling, zero-padding
- Feature vectors will be the same size, equal in size to the number of keypoints
- Feature vectors are generated using another model (from MediaPipe/DLIB) so depend on how good those models are
	- May not be reliable or more reliable
	- Data generated from data?
- Feature vectors will be generated independent of many biases
	- Independent of things such as skin colour, size, etc
	- Not independent of lip disfigurements but neither is image based
**Conclusion**:
- Image based would be initially more work for potentially worse results
- Keypoints are easier to generate, more reliable and have few biases for training off of
- Keypoints should be the first method, possible an extra step would be to also train using images and compare the results  

#### ##Frame Rate

#### Data Preprocessing

###### Normalisaation
- Data should be normalised. Currently keypoint information is a relative value of the position in frame
- This largely depends on where the person in the clip is, which will therefore skew any models produced
- Instead data should be normalised and made relative based off something else such as:
	- Another keypoint: Nose, eyes, etc
	- The mouth as a whole, like a mouth bounding box
- I decided to go with the **nose keypoint**:
	- Other keypoints could vary, the nose is very close to the mouth and consistently so
	- A bounding box made for the whole mouth could be affected by other things and maybe not be as reliable as nose keypoint
	- **To do this** the nose keypoint #3 will be used and subtracted from all other mouth keypoints
	- In the future **other normalisation methods** could be tried

###### Padding
- Typically data of different sizes is cut or padded in order to train with it
- As lip keypoints are being trained on, this is a static size. However, clip/word lengths are not
- Therefore each sample will be padded out to be 30 frames long
	- Cutting is not carried out, to preserve data
	- Keypoint lists filled with 0s will be used to pad
	- 30 is used as few words will be longer than 30 words and these are less likely to arise

#### Data Split
- Have the data LRW and LRS2 for lip reading
- Should use one for training models and the other for model testing on unseen data
	- Will show proper performance of the models: testing on data it has been trained (or tested/validated) against is less reliable
	- Need a substantial amount in order to test, using a whole dataset for this is therefore a good idea
- Which to use for model training/testing
	- LRW is 70GB
	- LRS2 is 50GB
- Should use LRS2 for training (as there’s less data) and LRW for testing
	- Model training data split
	- Will use 80/10/10 data split at first to see how this performs
	- Will adjust based on performance

#### Colab Pricing
- Colab comes with a free version, offering a basic GPU to run scripts
- This would instantly speed up training models and generating data and so should be used
- To use any data on Colab it must be uploaded to Google Drive
	- I decided to subscribe to get 2TB of storage
	- Costing can be found [here](https://one.google.com/about/plans?hl=en_GB)
	- I went with 2TB whilst this begins as only £2 a month and then increases to just £8 I though this was necessary
	- Extra storage is going to be needed for the amount of data

**Colab GPUs**
- There are multiple different plans for GPUs from Colab as can be found [here](https://colab.research.google.com/signup/pricing?utm_source=notebook_settings&utm_medium=link&utm_campaign=premium_gpu_selector) 
- I decided to experiment and see how far the free GPU could take model training and running data generation, to see if this is a viable option or not
- Options:
	- **"Pay As You Go"** would maybe suit me better. This allows me to only pay for the GPUs I use, therefore on months were model training isn't required as much this could be better
	- **"Colab Pro"** may also be a good choice, to limit how much may be spent in times when lots of training is required. It would be more consistent and stable so could be better. It is also as much as the lowest form of "Pay As You Go" when GPUs are being used
	- **"Colab Pro+"** would be too much. It's a huge investment and probably for little actual use of the GPUs

From this experimentation I discovered that the free GPU was good but model training could largely be enhanced using faster GPUs and background execution. Findings:
- Faster GPUs will be required to create any models of large size
- Background execution will also be needed
- More credits than the lower levels of Colab (like **"Pay As You Go"** and **"Colab Pro"**) provide, will be required for repeated experimentation and refinement
Therefore my decision was to use **"Pay As You Go"** each month, however as soon as further experimentation is required to increase to **"Colab Pro+"** in order to allow for the very useful background execution and faster GPUs.

Another thing I found was that Colab is very slow at **importing data** from Google Drive. Especially with the amount of data needed, this is a huge time sink for training anything. After doing [research](https://medium.com/@vishakha1203/easiest-way-to-upload-large-datasets-to-google-colab-1f89231844dc) I found that a common work around for this is to upload a zip to the Colab space and, each time data generation or model training is run, unzip this file to access data. This method still needs experimentation but seems viable.

Furthermore, the Uni provides [GPUs](https://ri.itservices.manchester.ac.uk/csf3/). This are slower and monitoring progress is hard. Therefore they would be better to use in later stages of the project, when models can be run without supervision. The beginning of the project is a major state of flux where many ideas are being tried out. Therefore the CSF, especially whilst it is currently not remotely available, is less useful.

**Conclusion**
- Use **"Pay As You Go"** version of Colab for now
- Increase to **"Colab Pro+"** when necessary
- Use the CSF when the project work becomes more stable

#### GUI Technology

Research found [here](https://www.activestate.com/blog/top-10-python-gui-frameworks-compared/).

Decided to go with Tkinter:
- Used this tool previously
- Easy to learn & use
- Most commonly used
- Not paid