import time
import os
import random
import requests
import base64
import io

from logid import generate
import openai
from memoization import cached
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from mimetypes import guess_type
from PIL.PngImagePlugin import PngImageFile

from data_gen.openai.base_model import Model
# from auto_caption.utils.file_utils import download_image

OPEN_API_KEY = os.environ.get('OPENAI_API_KEY')

def extract_frames(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    fps = vr.get_avg_fps()  # 获取视频的帧率
    # 按1fps抽取帧
    frame_idx = np.arange(0, total_frame_num, int(fps))
    # 确保不会超出总帧数
    frame_idx = np.clip(frame_idx, 0, total_frame_num - 1)
    frames = vr.get_batch(frame_idx).asnumpy()

    base64_frames = []
    for frame in frames:
        img = Image.fromarray(frame)
        output_buffer = io.BytesIO()
        img.save(output_buffer, format="JPEG")
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode("utf-8")
        img_base64 = {
            "url": f"data:image/jpeg;base64,{base64_str}", "detail": "high"}
        base64_frames.append(img_base64)
    return base64_frames


def PIL2b64(image: Image):
    output_buffer = io.BytesIO()
    image.save(output_buffer, format="PNG")
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode("utf-8")
    return base64_str

class GeminiModel(Model):
    # _cn_azure_endpoint = "https://search.bytedance.net/gpt/openapi/online/multimodal/crawl"
    # _global_azure_endpoint = "https://search-va.byteintl.net/gpt/openapi/online/multimodal/crawl"
    # _api_version = "2023-07-01-preview"
    _cache_max_size = 200

    def __init__(self, model_name='gptv', ak=None, log_prob=0.01, if_global=False, token_stat_percent=0.99, size_limit=None, 
                 cn_azure_endpoint=None, global_azure_endpoint=None, api_version=None) -> None:
        self.ak = OPEN_API_KEY if not ak else ak

        self.ak_state = {}
        self.ak_state_succ = {}
        self.log_prob = log_prob
        self.start_time = time.time()
        self.if_global = if_global
        self.size_limit = size_limit

        # self.cn_azure_endpoint = "https://search.bytedance.net/gpt/openapi/online/multimodal/crawl"
        # self.global_azure_endpoint = "https://search-va.byteintl.net/gpt/openapi/online/multimodal/crawl"
        # self.api_version = "2023-07-01-preview"

        self.cn_azure_endpoint = cn_azure_endpoint if cn_azure_endpoint else "https://search.bytedance.net/gpt/openapi/online/multimodal/crawl/openai/deployments/gpt_openapi"
        self.global_azure_endpoint = global_azure_endpoint if global_azure_endpoint else "https://search-va.byteintl.net/gpt/openapi/online/multimodal/crawl"
        self.api_version = api_version if api_version else "2024-03-01-preview"

        super().__init__(model_name, self.ak, token_stat_percent)

    def _creat_client(self, model_name, ak):
        client = openai.AzureOpenAI(
            azure_endpoint=self.global_azure_endpoint if self.if_global else self.cn_azure_endpoint,
            api_version=self.api_version,
            api_key=ak
        )
        client.temp_model_name = model_name
        client.temp_ak = ak

        # As model is a singleton, it's thread-safe. No need to lock.
        self.ak_state[ak[:5]] = 0
        self.ak_state_succ[ak[:5]] = 0
        return client

    @cached(max_size=_cache_max_size)
    def _format_image(self, image_url, to_base64, is_base64):
        if to_base64:
            try:
                # 本地转成base64
                image = 'data:image/jpeg;base64,' + \
                    download_image(image_url, to_base64=True)
            except Exception:
                # 如果失败了，再传url
                image = image_url
        else:
            if is_base64:
                image = 'data:image/jpeg;base64,'+image_url
            else:
                image = image_url
        return {'url': image, "detail": "high"}

    def _format_video(self, video_url, to_base64):
        if to_base64:
            try:
                if isinstance(video_url, PngImageFile):
                    video = PIL2b64(video_url)
                elif isinstance(video_url, bytes):
                    video = base64.b64encode(video_url).decode('utf-8')
                elif 'http' in video_url:
                    response = requests.get(video_url)
                    video = base64.b64encode(response.content).decode('utf-8')
                else:
                    mime_type, _ = guess_type(video_url)
                    with open(video_url, 'rb') as video_file:
                        video = base64.b64encode(
                            video_file.read()).decode('utf-8')
                        video = f"data:{mime_type};base64,{video}"
                    if self.size_limit is not None and len(video) > self.size_limit:
                        video = extract_frames(video_url)
                        return video
            except Exception:
                # 如果失败了，再传url
                video = video_url
        else:
            video = video_url
        return [{'url': video, "detail": "high"}]
    
    def _format_audio(self, audio_url, to_base64):
        if to_base64:
            try:
                if isinstance(audio_url, bytes):
                    audio = base64.b64encode(audio_url).decode('utf-8')
                elif 'http' in audio_url:
                    response = requests.get(audio_url)
                    audio = base64.b64encode(response.content).decode('utf-8')
                else:
                    mime_type, _ = guess_type(audio_url)
                    with open(audio_url, 'rb') as audio_file:
                        audio = base64.b64encode(
                            audio_file.read()).decode('utf-8')
                        audio = f"data:{mime_type};base64,{audio}"
            except Exception:
                # 如果失败了，再传url
                audio = audio_url
        else:
            audio = audio_url
        return [{'url': audio, "detail": "high"}]

    def __call__(self,
                 image_url,
                 prompt="What is the image about ?",
                 to_base64=False,
                 is_base64=False,
                 system_prompt=None,
                 max_tokens=4096,
                 return_output_token_length=False,
                 modality="video",
                 response_format=None,
                 thinking_budget=1024,
                 **kwargs):
        client = random.choice(self.clients)
        ak = client.temp_ak
        self.ak_state[ak[:5]] += 1
        if random.random() < self.log_prob:
            for ak in self.ak_state:
                print(
                    f"ak: {ak} 请求数：{self.ak_state[ak]}, 成功数：{self.ak_state_succ[ak]}, 成功率：{self.ak_state_succ[ak]/self.ak_state[ak]*100:.2f}%, 速度：{self.ak_state_succ[ak]/(time.time()-self.start_time)*60:.2f} 个/分钟")
            print(f"token_stat：{self.token_stat}")

        if not isinstance(image_url, list):
            image_url = [image_url]

        logid = generate()

        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})

        content_list = []

        vision_content = []
        for url in image_url:
            if modality == "image":
                vision_content.append(
                    self._format_image(url, to_base64, is_base64))
            elif modality == "video":
                vision_content.extend(self._format_video(url, to_base64))
            elif modality == 'audio':
                vision_content.extend(self._format_audio(url, to_base64))
        for v in vision_content:
            content_list.append({"type": "image_url", "image_url": v})

        content_list.append({
            "type": "text",
            "text": prompt
        })

        messages.append({
            "role": "user",
            "content": content_list,
        })

        if response_format is None:
            completion = client.chat.completions.create(
                model=client.temp_model_name,
                extra_headers={"X-TT-LOGID": logid},  # 请务必带上此header，方便定位问题
                messages=messages,
                max_tokens=max_tokens,
                timeout=360,
                extra_body={
                    "thinking": {
                        "include_thoughts": True,
                        "budget_tokens": thinking_budget
                    }
                }
                **kwargs
            )
        else:
            completion = client.chat.completions.create(
                model=client.temp_model_name,
                extra_headers={"X-TT-LOGID": logid},  # 请务必带上此header，方便定位问题
                messages=messages,
                max_tokens=max_tokens,
                timeout=360,
                response_format=response_format,
                extra_body={
                    "thinking": {
                        "include_thoughts": True,
                        "budget_tokens": thinking_budget
                    }
                },
                **kwargs
            )

        self.ak_state_succ[ak[:5]] += 1
        if self.token_stat_percent is not None:
            # 输出token统计
            if completion.usage.completion_tokens:
                self._update(completion.usage.completion_tokens)
        if return_output_token_length:
            # 允许输出token长度，便于筛选因输出达到max_tokens而被截断的数据
            return completion.choices[0].message.content, completion.usage.completion_tokens
        return completion.choices[0].message.content


if __name__ == '__main__':
    video_path = "/opt/tiger/lab/test.mp4"
    model = GeminiModel('gemini-1.5-pro-preview', ak='xxx')
    prompt = "<image>\nQuestion: What is the action performed by the person in the video?\nOptions:\n(A) Concealing something on top of something\n(B) Showing something on top of something\n(C) Not sure"
    print(model(video_path, prompt, to_base64=True,
                is_base64=False, modality="video"))
