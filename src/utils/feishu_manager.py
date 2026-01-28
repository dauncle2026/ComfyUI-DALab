import os
import logging
import requests
import torch
import numpy as np
import av
import io as python_io
import uuid
import urllib.parse
from PIL import Image, ImageOps

import folder_paths
import node_helpers
from comfy_api.latest import InputImpl

class FeishuManager:
    _instance = None
    _client = None
    _current_config_hash = None
    _config = None 
    _lark_module = None 

    def __new__(cls, config=None):
        if cls._instance is None:
            cls._instance = super(FeishuManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, config=None):
        if config:
            self._config = config
            
            app_id = config.get("app_id") or os.environ.get("FEISHU_APP_ID", "")
            app_secret = config.get("app_secret") or os.environ.get("FEISHU_APP_SECRET", "")

            new_hash = hash((app_id, app_secret))

            if not self._client or new_hash != self._current_config_hash:
                self._init_client(app_id, app_secret)
                self._current_config_hash = new_hash

    def _init_client(self, app_id, app_secret):
        try:
            import lark_oapi as lark
            self._lark_module = lark
        except ImportError:
            raise ImportError(
                "[DALab] Missing dependency: 'lark-oapi'. "
                "Please install it manually via terminal: 'pip install lark-oapi' "
                "(or 'python_embeded/python.exe -m pip install lark-oapi' if using portable ComfyUI)."
            )

        if not app_id or not app_secret:
            return

        try:
            self._client = lark.Client.builder() \
                .app_id(app_id) \
                .app_secret(app_secret) \
                .build()
            logging.info(f"[DALab] Feishu client initialized (AppID: {app_id[:6]}***).")
        except Exception as e:
            logging.error(f"[DALab] Client init failed: {e}")
            self._client = None
            raise

    @property
    def client(self):
        if not self._client:
            raise RuntimeError("[DALab] Feishu client not ready. Please check App ID/Secret in node config.")
        return self._client

    def fetch_and_process_data(self, app_token, table_id, view_id):
        lark = self._lark_module
        
        request = lark.api.bitable.v1.ListAppTableRecordRequest.builder() \
            .app_token(app_token) \
            .table_id(table_id) \
            .view_id(view_id) \
            .build()
        
        response = self.client.bitable.v1.app_table_record.list(request)
        if not response.success():
            raise Exception(f"[DALab] Fetch failed: {response.msg}")

        cfg = self._config
        open_key = cfg.get('open_field_name', 'open')
        open_val = cfg.get("open_field_value", "on")
        count_key = cfg.get("frame_count_field_name", "frame_count")

        type_configs = {
            "text":  self._parse_config_keys("text"),
            "image": self._parse_config_keys("image"),
            "audio": self._parse_config_keys("audio"),
            "video": self._parse_config_keys("video"),
        }

        results = {k: [[] for _ in range(v['count'])] for k, v in type_configs.items()}
        frame_counts = []
        record_ids = []

        for item in response.data.items:
            if item.fields.get(open_key, "") != open_val:
                continue

            frame_count = item.fields.get(count_key)
            frame_counts.append(int(frame_count) if frame_count else None)
            record_ids.append(item.record_id)

            for ftype, config_data in type_configs.items():
                keys = config_data['keys']
                for idx, key in enumerate(keys):
                    raw_val = item.fields.get(key, "" if ftype == "text" else {})
                    
                    if ftype == "text":
                        processed = raw_val
                    else:
                        processed = self._process_media_file(raw_val, ftype)
                    
                    results[ftype][idx].append(processed)

        return results["text"], results["image"], results["audio"], results["video"], frame_counts, record_ids

    def upload_media(self, image_tensor, app_token):
        lark = self._lark_module

        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor.squeeze(0)
            
        i_np = 255. * image_tensor.cpu().numpy()
        img_pil = Image.fromarray(np.clip(i_np, 0, 255).astype(np.uint8))
        
        img_resized = img_pil.copy()
        img_resized.thumbnail((256, 256), Image.LANCZOS)
        
        img_byte_arr = python_io.BytesIO()
        img_resized.save(img_byte_arr, format='JPEG', quality=80)
        file_size = img_byte_arr.tell()
        img_byte_arr.seek(0)

        request = lark.api.drive.v1.UploadAllMediaRequest.builder() \
            .request_body(lark.api.drive.v1.UploadAllMediaRequestBody.builder()
                .file_name(f"image_{uuid.uuid4()}.jpeg")
                .parent_type("bitable_image")
                .parent_node(app_token)
                .size(file_size)
                .file(img_byte_arr)
                .build()) \
            .build()

        response = self.client.drive.v1.media.upload_all(request)
        if not response.success():
            raise Exception(f"[DALab] Upload failed: {response.msg}")
        
        return response.data.file_token

    def batch_update_records(self, app_token, table_id, results, record_ids, field_name):
        lark = self._lark_module

        server_address = self._config.get("view_server_address")
        server_address = server_address.replace("http://", "").replace("https://", "").strip("/")
        records = list()
        for result, record_id in zip(results, record_ids):
            url_values = urllib.parse.urlencode(result)
            url = "http://{}/api/view?{}".format(server_address, url_values)
            records.append(lark.api.bitable.v1.AppTableRecord.builder().fields(
                {
                    field_name:{"link":"{}".format(url),"text":result["filename"]},
                }
            ).record_id(record_id).build())

        request = lark.api.bitable.v1.BatchUpdateAppTableRecordRequest.builder() \
            .app_token(app_token) \
            .table_id(table_id) \
            .user_id_type("open_id") \
            .request_body(lark.api.bitable.v1.BatchUpdateAppTableRecordRequestBody.builder()
                .records(records).build()) \
            .build()

        response = self.client.bitable.v1.app_table_record.batch_update(request)
        if not response.success():
            raise Exception(f"[DALab] Batch update failed: {response.msg}")
        
        return response.data

    def _parse_config_keys(self, ftype):
        opt_key = f"{ftype}_options"
        opts = self._config.get(opt_key, {})
        count_str = opts.get(opt_key, "[0]").replace("[", "").replace("]", "")
        count = int(count_str)
        keys = [opts.get(f"{ftype}{i+1}", f"{ftype}{i+1}") for i in range(count)]

        return {"count": count, "keys": keys}

    def _process_media_file(self, file_info, file_type):
        if not isinstance(file_info, dict): return None

        text_val = file_info.get("text", "").strip()
        link_val = file_info.get("link", "").strip()
        if not text_val and not link_val: return None

        filename, subfolder = self._resolve_filename(text_val, link_val)
        
        output_dir = folder_paths.get_output_directory()
        target_dir = os.path.join(output_dir, subfolder) if subfolder else output_dir
        local_path = os.path.join(target_dir, filename)

        if subfolder and not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)

        if not os.path.exists(local_path):
            if link_val:
                logging.info(f"[DALab] Downloading {file_type}: {filename}")
                if not self._download_url(link_val, local_path):
                    return None
            else:
                logging.warning(f"[DALab] Missing file & no link: {local_path}")
                return None

        try:
            if file_type == "image": return load_image_to_tensor(local_path)
            if file_type == "audio": return load_audio_to_tensor(local_path)
            if file_type == "video": return load_video_to_obj(local_path)
        except Exception as e:
            logging.error(f"[DALab] Error loading {file_type}: {e}")
            return None

    def _resolve_filename(self, text_val, link_val):
        def parse(url):
            if not url: return None, None
            try:
                p = urllib.parse.urlparse(url)
                if not p.netloc: return None, None
                q = urllib.parse.parse_qs(p.query)
                
                f = q.get("filename", [None])[0]
                s = q.get("subfolder", [None])[0]
                
                if not f and p.path: 
                    f = os.path.basename(p.path)
                return f, s
            except: 
                return None, None

        f, s = parse(text_val)
        if not f: f, s = parse(link_val)
        
        if not f: f = f"feishu_{uuid.uuid4().hex[:6]}.bin"
        if not s: s = ""
        
        f = f.replace("/", "_").replace("\\", "_").replace(":", "")
        return f, s

    def _download_url(self, url, path):
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            return True
        except Exception as e:
            logging.error(f"[DALab] Download failed: {e}")
            if os.path.exists(path): os.remove(path)
            return False

def load_video_to_obj(local_path):
    return InputImpl.VideoFromFile(local_path)

def load_audio_to_tensor(local_path):
    try:
        with av.open(local_path) as af:
            if not af.streams.audio:
                raise ValueError("[DALab] No audio stream found.")
            stream = af.streams.audio[0]
            sr = stream.codec_context.sample_rate
            n_channels = stream.channels
            frames = []
            for frame in af.decode(streams=stream.index):
                buf = torch.from_numpy(frame.to_ndarray())
                if buf.shape[0] != n_channels:
                    buf = buf.view(-1, n_channels).t()
                frames.append(buf)
            if not frames:
                raise ValueError("[DALab] No audio frames decoded.")
            wav = torch.cat(frames, dim=1)
            wav = f32_pcm(wav)
            return {"waveform": wav.unsqueeze(0), "sample_rate": sr}
    except Exception as e:
        logging.error(f"[DALab] Audio load error: {e}")
        return None

def load_image_to_tensor(image_path):
    try:
        img = node_helpers.pillow(Image.open, image_path)
    except Exception as e:
        logging.error(f"[DALab] Image load error: {e}")
        return None

    is_animated = getattr(img, 'is_animated', False)
    n_frames = getattr(img, 'n_frames', 1)
    if is_animated and n_frames > 1:
        img.seek(0)

    img = node_helpers.pillow(ImageOps.exif_transpose, img)
    if img.mode == 'I':
        img = img.point(lambda i: i * (1 / 255))
    
    image = img.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    return image

def f32_pcm(wav: torch.Tensor) -> torch.Tensor:
    if wav.dtype.is_floating_point:
        return wav
    elif wav.dtype == torch.int16:
        return wav.float() / 32768.0
    elif wav.dtype == torch.int32:
        return wav.float() / 2147483648.0
    return wav