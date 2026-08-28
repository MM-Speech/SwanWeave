import copy
import os
import sys
import json
from typing import List, Optional, Dict
import tempfile
import bytedtos
import base64

import os

# 配置项

import tempfile
from typing import Dict, Optional

import os
import sys
from glob import glob
import bytedtos
from pathlib import Path

import yaml


def send_file_to_tos(image,
                     sub_dir="gokuhuman_dpo",
                     bucket='sa-ag-sg-research-sg',
                     accessKey="*",
                     output='.output.txt',
                     return_type="offline",
                     image_name=None):
    if bucket == 'sa-ag-sg-research-sg':
        ak = '*'
        psm = 'toutiao.tos.tosapi'
        cluster = 'default'
        idc = 'sg1'
    if bucket == 'videoclip-embeddings-512d-sg':
        ak = '*'
        psm = 'toutiao.tos.tosapi'
        cluster = 'default'
        idc = 'sg1'

    tos_client = bytedtos.Client(
        bucket, ak, service=psm, cluster=cluster, idc=idc,
        connect_timeout=600, timeout=600)
    if image_name is None:
        image_name = "_".join(image.replace(" ", "").replace("#", "").split("/")[2:])
    obj = "{}/{}".format(sub_dir, image_name)
    link = "https://tosv-sg.tiktok-row.org/obj/{0}/{1}".format(bucket, obj)
    link_offline = "https://tosv-sg.tiktok-row.org/obj/{0}/{1}".format(bucket, obj)
    link_tos = "tos://{0}/{1}".format(bucket, obj)
    try:
        image_body = open(image, 'rb').read()
        tos_client.put_object(obj, image_body)
        ret = tos_client.head_object(obj)
        if return_type == "tos_dir":
            return link_tos
        elif return_type == "tos":
            return link
        else:
            return link_offline
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError("upload material {} error, errors: {}".format(image, e))
        return None


def gen_html(infos: Dict[str, list], names: Optional[List[str]] = None, output_fp: Optional[str] = None,
             title_name=None, extra_desc=None):
    if output_fp is None:
        output_fp = tempfile.NamedTemporaryFile(suffix=".html").name

    keys = list(infos.keys())
    if names is None:
        names = keys
    else:
        assert len(names) == len(keys), "The length of 'names' must match the number of files."

    with open(output_fp, 'w') as f:
        print('<html lang="en">', file=f)
        print('<head>', file=f)
        print('<meta charset="UTF-8">', file=f)
        print('<meta name="viewport" content="width=device-width, initial-scale=1.0">', file=f)
        print('<style>', file=f)
        print('''
            body { margin: 0; padding: 20px; font-family: Arial, sans-serif; }
            .container { max-width: 1280px; margin: 0 auto; }
            table { 
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }
            h1 {
                  font-size: 2em;
                  margin-bottom: 0.2em;
                }
            p.description {
                  color: #666;
                  margin-bottom: 1.5em;
            }
            th, td { 
                padding: 10px;
                border: 2px solid DodgerBlue;
                text-align: center;
            }
            th {
                background-color: #f0f8ff;
                font-weight: bold;
            }
            .video-container {
                width: 100%;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 405px; /* 保持占位高度 */
                background: #f5f5f5;
            }
            video {
                width: 100%;
                max-width: 720px;
                height: auto;
                max-height: 405px;
            }
            @media (max-width: 768px) {
                body { padding: 10px; }
                th, td { padding: 5px; }
                .video-container { min-height: 203px; } /* 移动端高度减半 */
            }
        ''', file=f)
        print('</style>', file=f)
        print('</head>', file=f)
        print('<body>', file=f)
        if title_name is not None:
            print(f'  <h1>{title_name}</h1>', file=f)
        if extra_desc is not None:
            print(f'  <p class="description">{extra_desc}</p>', file=f)
        print('<div class="container">', file=f)
        print('<table>', file=f)
        print('<thead><tr>', file=f)
        for name in names:
            print(f'<th>{name}</th>', file=f)
        print('</tr></thead>', file=f)
        print('<tbody>', file=f)

        num_rows = min(len(infos[k]) for k in keys)
        num_rows = min(num_rows, 100)

        for idx in range(num_rows):
            print('<tr>', file=f)
            for k in keys:
                value = infos[k][idx].strip()
                if value.startswith(('http://', 'https://')):
                    video_src = value
                    if '?' in video_src:
                        video_src += '&autoplay=0'
                    else:
                        video_src += '?autoplay=0'

                    print('<td>', file=f)
                    print('<div class="video-container">', file=f)
                    print(f'<video class="lazy-video" controls muted playsinline data-src="{video_src}">', file=f)
                    print('您的浏览器不支持 video 标签。', file=f)
                    print('</video>', file=f)
                    print('</div>', file=f)
                    print('</td>', file=f)
                else:
                    print(f'<td>{value}</td>', file=f)
            print('</tr>', file=f)

        print('</tbody>', file=f)
        print('</table>', file=f)
        print('</div>', file=f)

        # 添加懒加载脚本
        print('''
        <script>
            document.addEventListener("DOMContentLoaded", function() {
                var lazyVideos = [].slice.call(document.querySelectorAll("video.lazy-video"));

                if ("IntersectionObserver" in window) {
                    var lazyVideoObserver = new IntersectionObserver(function(entries, observer) {
                        entries.forEach(function(entry) {
                            if (entry.isIntersecting) {
                                var video = entry.target;
                                var videoSrc = video.dataset.src;

                                // 创建source元素并设置src
                                var source = document.createElement('source');
                                source.src = videoSrc;
                                source.type = 'video/mp4';
                                video.appendChild(source);

                                video.classList.remove("lazy-video");
                                lazyVideoObserver.unobserve(video);
                            }
                        });
                    });

                    lazyVideos.forEach(function(lazyVideo) {
                        lazyVideoObserver.observe(lazyVideo);
                    });
                }
            });
        </script>
        ''', file=f)
        print('</body>', file=f)
        print('</html>', file=f)
    return output_fp


def gen_audio_html(infos: Dict[int, dict], output_fp: Optional[str] = None,
                   title_name=None, extra_desc=None):
    if output_fp is None:
        output_fp = tempfile.NamedTemporaryFile(suffix=".html", delete=False).name

    num_per_row = 5
    total = len(infos)
    rows = (total + num_per_row - 1) // num_per_row

    with open(output_fp, 'w') as f:
        print('<html lang="en">', file=f)
        print('<head>', file=f)
        print('<meta charset="UTF-8">', file=f)
        print('<meta name="viewport" content="width=device-width, initial-scale=1.0">', file=f)
        print('<style>', file=f)
        print('''
            body { margin: 0; padding: 20px; font-family: Arial, sans-serif; }
            .container { max-width: 1280px; margin: 0 auto; }
            table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
            h1 {
                  font-size: 2em;
                  margin-bottom: 0.2em;
                }
            p.description {
                  color: #666;
                  margin-bottom: 1.5em;
            }
            td { padding: 10px; border: 2px solid DodgerBlue; vertical-align: top; text-align: center; }
            audio { width: 100%; }
            .desc { margin-top: 10px; white-space: pre-wrap; text-align: left; font-size: 14px; background: #f8f8f8; padding: 8px; border-radius: 5px; }
        ''', file=f)
        print('</style>', file=f)
        print('</head>', file=f)
        print('<body>', file=f)
        if title_name is not None:
            print(f'  <h1>{title_name}</h1>', file=f)
        if extra_desc is not None:
            print(f'  <p class="description">{extra_desc}</p>', file=f)
        print('<div class="container">', file=f)
        print('<table>', file=f)

        keys = list(infos.keys())

        for row in range(rows):
            print('<tr>', file=f)
            for col in range(num_per_row):
                idx = row * num_per_row + col
                if idx >= total:
                    print('<td></td>', file=f)
                    continue
                info = infos[idx]
                tos_url = info.get('tos_url', '')
                local_prompt = info.get('local_prompt', '')
                text = info.get('text', '')
                global_prompt = info.get('global_prompt', '')

                print('<td>', file=f)
                print(f'<audio controls preload="none">', file=f)
                print(f'  <source src="{tos_url}" type="audio/mpeg">', file=f)
                print('  Your browser does not support the audio element.', file=f)
                print('</audio>', file=f)
                print('<div class="desc">', file=f)
                print(f'Local Prompt: {local_prompt.replace("<", "&lt;").replace(">", "&gt;")}\n', file=f)
                print(f'Text: {text}\n', file=f)
                print(f'Global Prompt: {global_prompt}', file=f)
                print('</div>', file=f)
                print('</td>', file=f)
            print('</tr>', file=f)

        print('</table>', file=f)
        print('</div>', file=f)
        print('</body>', file=f)
        print('</html>', file=f)

    return output_fp

def main(yml=None, out_path=None, title_name=None, extra_desc=None):
    sub_dir = 'scriptspeech_gqj'
    if yml is None:
        yml = './egs/inference/inference_na_sample0728.yaml'
    if out_path is None:
        out_path = './infer_out/tts/250625_scriptspeech_semanticlm_singlespk_01_250801_scriptspeech_dit_s2_c_vm_1_20250802_144403'

    from collections import defaultdict
    infos = defaultdict(dict)

    with open(yml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        samples = cfg['samples']
    files = glob(f'{out_path}/*.wav')
    assert len(files) == len(samples)
    for idx, sample in enumerate(samples):
            wav = os.path.join(out_path, f"out_{idx}.wav")
            tos_url = send_file_to_tos(wav, sub_dir=sub_dir)
            # tos_url = f"https://tosv-sg.tiktok-row.org/obj/sa-ag-sg-research-sg/scriptspeech_gqj/tts_250625_scriptspeech_semanticlm_singlespk_01_250723_scriptspeech_dit_s2_c_1_20250724_070952_out_{idx}.wav"
            print("tos_url: ", tos_url)
            infos[idx]['tos_url'] = tos_url
            infos[idx].update(sample)

    # 调用gen_html函数时传递names参数
    html_path = gen_audio_html(infos, title_name=title_name, extra_desc=extra_desc)
    print(f"生成的HTML文件路径：{html_path}")
    html_tos = send_file_to_tos(html_path, sub_dir=sub_dir)
    print(f"生成的HTML文件路径：{html_tos}")


if __name__ == "__main__":
    main()