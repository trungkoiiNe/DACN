from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
import time

def take_photo(filename='photo.jpg', quality=0.8):
    js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      
      try {
        console.log('Requesting camera access...');
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "user",
            width: { ideal: 1280 },
            height: { ideal: 720 }
          }
        });
        console.log('Camera access granted');

        document.body.appendChild(div);
        div.appendChild(video);
        video.srcObject = stream;
        
        // Add a small delay before playing
        await new Promise(resolve => setTimeout(resolve, 1000));
        await video.play();

        google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

        await new Promise((resolve) => capture.onclick = resolve);

        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        stream.getVideoTracks()[0].stop();
        div.remove();
        return canvas.toDataURL('image/jpeg', quality);
      } catch (err) {
        div.remove();
        throw new Error('Error accessing camera: ' + err.message);
      }
    }
    ''')
    
    print("Initializing camera...")
    display(js)
    
    try:
        data = eval_js('takePhoto({})'.format(quality))
        binary = b64decode(data.split(',')[1])
        with open(filename, 'wb') as f:
            f.write(binary)
        return filename
    except Exception as err:
        print(f"Error: {str(err)}")
        print("Please make sure to:")
        print("1. Grant camera permissions when prompted")
        print("2. Use Google Chrome browser")
        print("3. Check if your camera is working properly")
        raise
