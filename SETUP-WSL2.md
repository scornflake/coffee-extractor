# WSL2 - Windows

Install WSL2 via Windows Store

Install Ubuntu 22.04 (via Windows Store is OK)

The reference for the above was: https://www.tensorflow.org/install/pip#windows-wsl2

Check out the source code in an Ubuntu (I used 22.04) environment.

Prepare to be able to build the WHL deps as required, by: 

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv python3-dev gcc
sudo apt-get install -y build-essential portaudio19-dev tesseract-ocr ffmpeg python3-tk libgl1-mesa-glx
```

You are ready to go when this command works:

```bash
pip install pyaudio
```

Now you can pip install the requirements-wsl2.txt file - the main difference here is that
tensorflow gets installed with the CUDA support for WSL2, and I've added a few other modules (pytesseract).  
 
So it's off to the races we go with the pip install:
```bash
pip install -r requirements-wsl2.txt
```

# TkInter

FYI ONLY: the fix here is included in the above apt install command. 

```bash
sudo apt-get install python3-tk
```

# OpenCV / libGL.so.1

FYI ONLY: the fix here is included in the above apt install command.

ImportError: libGL.so.1: cannot open shared object file: No such file or directory

```bash
sudo apt-get install libgl1-mesa-glx
```

# Editor - Tessaract

The tesseract OCR command is installed via sudo apt get, and the python package via the requirements-wsl2.txt file.  

To run the Editor, I used this: 

```bash
export TESSDATA_PREFIX=../../src/coffee-extractor/teseract/tessdata
cd ./copy-of-google/1
python ../../src/coffee-extractor/editor.py input.spec.json
```

# Copy Google Data

Google Drive syncs a .LNK file to the Windows file system to represent the "My Drive" folder.

There is no way to traverse a Windows .LNK file in WSL2, you will need to copy the 
Google data *into* your WSL2 environment.  

I literally copy/pasted the entire Coffee AI directory into the home folder of my WSL2 user.

Not ideal, but I didn't want to spend time on this.
