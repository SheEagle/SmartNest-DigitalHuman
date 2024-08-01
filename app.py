from flask import Flask

app = Flask(__name__)

import os
import random
from flask import request, jsonify, send_file
from LLM import LLM
from ASR import FunASR
from TFG import SadTalker
from TTS import EdgeTTS
from VITS import GPT_SoVITS
from src.cost_time import calculate_time
import datetime
from moviepy.video.io.VideoFileClip import VideoFileClip

# 设置全局变量
mode = 'offline'
llm = LLM(mode=mode).init_model('Qwen', 'Qwen/Qwen-1_8B-Chat')
talker = SadTalker(lazy_load=True)
asr = FunASR()
tts = EdgeTTS()
vits = GPT_SoVITS()

# 设定默认参数值，可修改
source_image = r'image.jpg'
blink_every = True
size_of_image = 256
preprocess_type = 'crop'
facerender = 'facevid2vid'
enhancer = False
is_still_mode = False
pic_path = "boy.png"
crop_pic_path = "./inputs/first_frame_dir_boy/boy.png"
first_coeff_path = "./inputs/first_frame_dir_boy/boy.mat"
crop_info = ((876, 747), (0, 0, 886, 838), [10.382158280494476, 0, 886, 747.7078990925525])

exp_weight = 1

use_ref_video = False
ref_video = None
ref_info = 'pose'
use_idle_mode = False
length_of_audio = 5


@calculate_time
def LLM_response(question_audio, question, voice='zh-CN-XiaoxiaoNeural', rate=0, volume=0, pitch=0):
    answer = llm.generate(question)
    # answer = "您好，请问有什么能帮到您的呢？"
    print(answer)
    voice = 'zh-CN-XiaoxiaoNeural'
    if voice in tts.SUPPORTED_VOICE:
        try:
            tts.predict(answer, voice, rate, volume, pitch, 'answer.wav', 'answer.vtt')
        except:
            os.system(f'edge-tts --text "{answer}" --voice {voice} --write-media answer.wav')
    elif voice == "克隆烟嗓音":
        gpt_path = "../GPT-SoVITS/GPT_weights/yansang-e15.ckpt"
        sovits_path = "../GPT-SoVITS/SoVITS_weights/yansang_e16_s144.pth"
        vits.load_model(gpt_path, sovits_path)
        vits.predict(ref_wav_path="examples/slicer_opt/vocal_output.wav_10.wav_0000846400_0000957760.wav",
                     prompt_text="你为什么要一次一次的伤我的心啊？",
                     prompt_language="中文",
                     text=answer,
                     text_language="中英混合",
                     how_to_cut="按标点符号切",
                     save_path='answer.wav')
    elif voice == "克隆声音":
        if question_audio is None:
            print("无声音输入，无法克隆声音")
            return None, None, None
        gpt_path = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
        sovits_path = "GPT_SoVITS/pretrained_models/s2G488k.pth"
        vits.load_model(gpt_path, sovits_path)
        vits.predict(ref_wav_path=question_audio,
                     prompt_text=question,
                     prompt_language="中文",
                     text=answer,
                     text_language="中英混合",
                     how_to_cut="凑四句一切",
                     save_path='answer.wav')
    return 'answer.wav', None, answer


@calculate_time
def Talker_response(question_audio, text, voice='zh-CN-XiaoxiaoNeural', rate=0, volume=100, pitch=0, batch_size=2):
    # voice = 'zh-CN-XiaoxiaoNeural' if voice not in tts.SUPPORTED_VOICE else voice
    # print(voice , rate , volume , pitch)
    driven_audio, driven_vtt, _ = LLM_response(question_audio, text, voice, rate, volume, pitch)
    pose_style = random.randint(0, 45)
    # video = talker.test(pic_path,
    #                     crop_pic_path,
    #                     first_coeff_path,
    #                     crop_info,
    #                     source_image,
    #                     driven_audio,
    #                     preprocess_type,
    #                     is_still_mode,
    #                     enhancer,
    #                     batch_size,
    #                     size_of_image,
    #                     pose_style,
    #                     facerender,
    #                     exp_weight,
    #                     use_ref_video,
    #                     ref_video,
    #                     ref_info,
    #                     use_idle_mode,
    #                     length_of_audio,
    #                     blink_every,
    #                     fps=20)
    video = talker.test2(source_image,
                         driven_audio,
                         preprocess_type,
                         is_still_mode,
                         enhancer,
                         batch_size,
                         size_of_image,
                         pose_style,
                         facerender,
                         exp_weight,
                         use_ref_video,
                         ref_video,
                         ref_info,
                         use_idle_mode,
                         length_of_audio,
                         blink_every,
                         fps=20)

    # print(type(video))

    if driven_vtt:
        return video, driven_vtt
    else:
        return video


def generate_video(video_path):
    with open(video_path, 'rb') as video_file:
        while True:
            video_chunk = video_file.read(1024)
            if not video_chunk:
                break
            yield video_chunk



def extract_audio(input_video, output_audio):
    video_clip = VideoFileClip(input_video)
    audio_clip = video_clip.audio

    audio_clip.write_audiofile(output_audio)


@app.route('/audio', methods=['POST'])
def generate_video():
    try:
        question_video = request.files['video']

        # 获取当前日期
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        base_video_directory = 'video'
        base_audio_directory = 'audio'
        # 在保存目录后面加一层以日期命名的文件夹
        save_video_directory = os.path.join(base_video_directory, current_date)
        save_audio_directory = os.path.join(base_audio_directory, current_date)
        # 创建目录（如果不存在）
        os.makedirs(save_video_directory, exist_ok=True)
        os.makedirs(save_audio_directory, exist_ok=True)
        # 构建保存路径（日期+序号）
        file_number = 1
        audio_path = os.path.join(save_audio_directory, f"{file_number}.wav")
        video_path = os.path.join(save_video_directory, f"{file_number}.mp4")
        # 确保路径不会覆盖已存在的文件
        while os.path.exists(audio_path):
            file_number += 1
            audio_path = os.path.join(save_audio_directory, f"{file_number}.wav")
        while os.path.exists(video_path):
            file_number += 1
            video_path = os.path.join(save_video_directory, f"{file_number}.mp4")

        question_video.save(video_path)
        extract_audio(video_path, audio_path)

        # convert_3gp_to_wav(audio_3gp_path,audio_wav_path)

        input_text = asr.transcribe(audio_path)


        voice = 'zh-CN-XiaoxiaoNeural'


        rate = 5
        volume = 20
        pitch = 5

        video_path = Talker_response(audio_path, input_text, voice, rate, volume, pitch)

        # 返回生成的视频文件
        return send_file(video_path, as_attachment=True)

    except Exception as e:
        # 返回 JSON 格式的错误响应
        error_response = {
            'status': 'error',
            'message': str(e),
        }
        return jsonify(error_response), 500  # 500 表示服务器内部错误


@app.route('/image', methods=['POST'])
def upload_image():
    # 检查是否有上传的文件
    if 'image' not in request.files:
        return jsonify({'error': '未上传图片！'}), 400

    # 获取上传的视频文件
    image = request.files['image']

    # 检查文件是否为空
    if image.filename == '':
        return jsonify({'error': '图片为空！'}), 400

    try:
        image.save('image.jpg')
        success_response = {
            'status': 'success',
            'message': '图片上传成功！',
        }
        return jsonify(success_response), 200  # 200 表示成功

    except Exception as e:
        # 返回 JSON 格式的错误响应
        error_response = {
            'status': 'error',
            'message': str(e),
        }
        return jsonify(error_response), 500  # 500 表示服务器内部错误


if __name__ == '__main__':
    app.run(port=5000)
