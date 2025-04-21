import pyttsx3
import pyaudio
import wave
import threading
import os

def text_to_speech(text, filename):
    # 初始化TTS引擎
    engine = pyttsx3.init()
    
    # 设置语速和音量（可选）
    engine.setProperty('rate', 200)  # 语速
    engine.setProperty('volume', 1)  # 音量

    # 保存音频到文件
    engine.save_to_file(text, filename)
    engine.runAndWait()

def play_audio(filename):
    # 打开WAV文件
    wf = wave.open(filename, 'rb')

    # 初始化PyAudio
    p = pyaudio.PyAudio()

    # 打开音频流
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # 读取和播放音频流
    chunk = 4096  # 增大块大小以减少延迟
    data = wf.readframes(chunk)
    i=0
    while data:
        stream.write(data)
        data = wf.readframes(chunk)
        i=i+1
        print(i)

    # 关闭流
    stream.stop_stream()
    stream.close()

    # 关闭PyAudio
    p.terminate()

def play_audio_async(filename):
    thread = threading.Thread(target=play_audio, args=(filename,))
    thread.start()

if __name__ == '__main__':
    # 示例文本
    text = "mouse"

    # 生成音频文件
    audio_filename = os.path.join(os.path.expanduser("~"), "Desktop", "mouse.wav")
    text_to_speech(text, audio_filename)
    print("Audio file saved.")

    # 异步播放音频文件
    play_audio_async(audio_filename)
    print("Audio playback started.")
