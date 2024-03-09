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

# CUDA on WSL2

Get CUDA drivers for WSL2 on Windows. 

Ref: https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html#wsl

```bash
sudo apt-key del 7fa2af80
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-3
```

The throw this into your .bashrc file:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

# Set up venv (OR similar)

```bash
cd $HOME
python3 -m venv venv  
source venv/bin/activate
```

# Get the code

```bash
git config --global user.name "John Clayton"
git config --global user.email johnclayton72@gmail.com
```

# If using 1Password SSH Integration with WSL

To make sure Git within WSL uses the Windows Git, and therefore also the 1Password SSH integration, run
this command:

```bash
git config --global core.sshCommand ssh.exe
```

# Python Requirements 

Do this first, if it works its like validating all the stuff above.

Tip: make sure you activate your venv first.

```bash
pip install pyaudio
```

Now you can pip install the requirements-wsl2.txt file - the main difference here is that
tensorflow gets installed with the CUDA support for WSL2, and I've added a few other modules (pytesseract).  
 
So it's off to the races we go with the pip install:
```bash
pip install -r requirements-wsl2.txt
```

# Copy Google Data

Google Drive syncs a .LNK file to the Windows file system to represent the "My Drive" folder.

There is no way to traverse a Windows .LNK file in WSL2, you will need to copy the 
Google data *into* your WSL2 environment.  

I literally copy/pasted the entire Coffee AI directory into the home folder of my WSL2 user.

Not ideal, but I didn't want to spend time on this.

# Run Editor - Tesseract

The tesseract OCR command is installed via sudo apt get, and the python package via the requirements-wsl2.txt file.  

To run the Editor, I used this: 

```bash
export TESSDATA_PREFIX=../../src/coffee-extractor/tesseract/tessdata
cd ./copy-of-google/1
python ../../src/coffee-extractor/editor.py input.spec.json
```

# Reference

## TkInter

FYI ONLY: the fix here is included in the above apt install command. 

```bash
sudo apt-get install python3-tk
```

## OpenCV / libGL.so.1

FYI ONLY: the fix here is included in the above apt install command.

ImportError: libGL.so.1: cannot open shared object file: No such file or directory

```bash
sudo apt-get install libgl1-mesa-glx
```

