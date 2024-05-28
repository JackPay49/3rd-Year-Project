# 3rd Year Project
My 3rd year project, or dissertation. This project researched Deep Learning for the application of lip reading. Over the course of 9 months I developed a data generation pipeline, trained various lip reading models, created a graphical user interface (GUI), filmed a short presentation, and wrote a 14,000 word dissertation analysing my work.

This project was key towards learning more about Keras, Tkinter, Deep Learning and Computer Vision.

## Project Stages
- **Data Generation**: During this stage I was given permission by the BBC to use the dataset [Lip Reading in the Wild (LRW)](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html). I then used Python & MediaPipe to create a data generation pipeline. This was an automated process to collect data for training lip reading models. Data came in the form of both image crops and lip landmark coordiantes of people's lips
- **Model Training**: Next, using this data various experimentation was carried out into producing the best performing lip reading model. I experimented with LSTMs, basic transformers, CNNs, CTC loss and more. In total 17 experiments were conducted, with the best model achieving a test accuracy and loss of 97% and 0.0949
- **GUI Development**: A basic GUI was developed using Python & Tkinter to showcase the different models in real-time and configure how the models were employed. A finetuning section was added to the GUI allowing models to be tuned towards specific speakers
- **Dissertation**: With development complete, I then wrote my dissertation. Totalling 14,272 words and 102 pages, my dissertation looked into the background of lip reading & machine learning, analysed my training methods & Deep learning models, and presented my GUI
- **Screencast**: Finally, a short, 7 minute, presentation was filmed. This quickly explained and showcased my work

## Quick-Links
- [Dissertation](https://github.com/JackPay49/3rd-Year-Project/blob/main/dissertation/current%20draft.pdf)
- [Screencast](https://drive.google.com/drive/folders/1m4s5uxAoGdJQ2NH0BvSIRnIa34fncyxe)
- [GUI Code](https://github.com/JackPay49/3rd-Year-Project/blob/main/classes/graphical/run_gui.py)
