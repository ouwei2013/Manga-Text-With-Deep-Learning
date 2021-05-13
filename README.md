
一个从漫画中自动识别对话框的项目。为识别对话框，采用了深度学习与传统图像处理方法结合（基于mser的边缘识别）的策略，效果不错。具体原理请参照document文件
# Manga-Text-With-Deep-Learning

>  Based on the work Created by [Walter Betini](https://github.com/walterBSG) and [Victor Guedes](https://github.com/VictorGuedes)

# How to use

 1. In the text_detector.py file, call the detect_chars_rect function. (Please make sure the initializer function is called to initialize the tensorflow graph before the detect_chars_rect function is called. )
 2. detect_chars_rect function resutns a list of boxes. Each box is represented by (slice(y_start,y_stop),slice(x_start,x_stop)).  It can be transformed into the traditional (x,y,w,h) format by: (box[1].start, box[0].start, box[1].stop-box[1].start, box[0].stop-box[0].start)
 3.  In command line window, issue python text_detector.py my_file_path, it will show the detection result.
 4. example.py provides an extra example about how to use it to process a directory hosting multiple images


 #  Underlying Algorithms
 > The text_detector consists of two models: deep learning model, non-deep learning model. The non-deep learning model firsts detects all the edges in an image, then filter out edges that have low possibilities of containing text. I combine the results of deep-learning model and the edge detection based model as the final results.  For details, please check the 'how_to_work' file.


## TODOs
- [ ] Filter out  non-text boxes

## Requirements
  ```
pip install tensorflow-gpu
pip install keras
pip install scipy
pip install matplotlib
pip install opencv-python
pip install pytesseract
```
