import os
import tos
import logging
import traceback

import tenacity

logger = logging.getLogger(__name__)
min_part_size = 100 * 1024 * 1024

"""
https://www.volcengine.com/docs/6349/92785
"""

class VolcTosClient:
    def __init__(self, bucket='video-data-downloader', *, ak=None, sk=None, endpoint=None, region=None):
        self.bucket_name = bucket

        env_bucket = bucket.upper().replace('-', '_')
        env_prefix = f"VOLC_TOS_{env_bucket}"  # e.g. VOLC_TOS_VIDEO_DATA_DOWNLOADER_AK

        ak = ak or os.environ.get(f"{env_prefix}_AK")
        sk = sk or os.environ.get(f"{env_prefix}_SK")
        endpoint = endpoint or os.environ.get(f"{env_prefix}_ENDPOINT") or "tos-cn-beijing.volces.com"
        region = region or os.environ.get(f"{env_prefix}_REGION") or "cn-beijing"

        missing = []
        if not ak:
            missing.append(f"{env_prefix}_AK")
        if not sk:
            missing.append(f"{env_prefix}_SK")
        if missing:
            raise ValueError("Missing Volc TOS credentials. Set env vars: " + ", ".join(missing))

        self.client = tos.TosClientV2(ak, sk, endpoint, region)

    def normalize_k(self, k: str):
        if k.startswith('tos://'):
            prefix = f"tos://{self.bucket_name}/"
            if k.startswith(prefix):
                k = k[len(prefix):]
        k = k.replace('//', '/')
        return k

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=8),
        after=tenacity.after_log(logger, logging.WARN))
    def put_object(self, k, v):
        tos_client = self.client
        bucket_name = self.bucket_name
        k = self.normalize_k(k)
        total_size = len(v)
        if total_size < min_part_size:
            try:
                resp = tos_client.put_object(bucket_name, k, content=v)
                return resp
            except:
                traceback.print_exc()
                raise
        else:
            multi_result = tos_client.create_multipart_upload(bucket_name, k, acl=tos.ACLType.ACL_Public_Read,
                                                  storage_class=tos.StorageClassType.Storage_Class_Standard)
            upload_id = multi_result.upload_id
            parts = []

            part_number = 1
            offset = 0
            while offset < total_size:
                upload_size = min(min_part_size, total_size - offset)
                resp = self.upload_part(k, upload_id, part_number, v[offset: offset + upload_size], offset)
                parts.append(resp)
                offset += upload_size
                part_number += 1
            
            resp = tos_client.complete_multipart_upload(bucket_name, k, upload_id, parts)
            return resp

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=8),
        after=tenacity.after_log(logger, logging.WARN))
    def upload_part(self, key: str, upload_id: str, part_number: str, data: bytes, offset):
        tos_client = self.client
        bucket_name = self.bucket_name
        try:
            return tos_client.upload_part(bucket_name, key, upload_id, part_number, content=data, init_offset=offset)
        except:
            traceback.print_exc()
            raise

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=8),
        after=tenacity.after_log(logger, logging.WARN))
    def upload_file(self, k, path, task_num=3, part_size=1024 * 1024 * 5):
        tos_client = self.client
        bucket_name = self.bucket_name
        k = self.normalize_k(k)
        try:
            tos_client.upload_file(bucket_name, k, path, task_num=task_num, part_size=part_size)
        except:
            traceback.print_exc()
            raise

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=8),
        after=tenacity.after_log(logger, logging.WARN))
    def get_object(self, k):
        tos_client = self.client
        bucket_name = self.bucket_name
        k = self.normalize_k(k)
        try:
            object_stream = tos_client.get_object(bucket_name, k)
            data = object_stream.read()
            return data
        except tos.exceptions.TosServerError as e:
            print('fail with server error, code: {}'.format(e.code))
            traceback.print_exc()
            raise
        except:
            traceback.print_exc()
            raise

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=8),
        after=tenacity.after_log(logger, logging.WARN))
    def download_file(self, k, path, task_num=3, part_size=1024 * 1024 * 5):
        tos_client = self.client
        bucket_name = self.bucket_name
        k = self.normalize_k(k)
        try:
            tos_client.download_file(bucket_name, k, path, task_num=task_num, part_size=part_size)
        except:
            traceback.print_exc()
            raise

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=8),
        after=tenacity.after_log(logger, logging.WARN))
    def list_prefix(self, prefix):
        tos_client = self.client
        bucket_name = self.bucket_name
        prefix = self.normalize_k(prefix)
        try:
            truncated = True
            continuation_token = ''
            ks = []
            while truncated:
                result = tos_client.list_objects_type2(bucket_name, prefix=prefix, continuation_token=continuation_token)
                for item in result.contents:
                    ks.append(item.key)
                truncated = result.is_truncated
                continuation_token = result.next_continuation_token
            return ks
        except:
            traceback.print_exc()
            raise

    def check_tos_file_exists(self, k):
        tos_client = self.client
        bucket_name = self.bucket_name
        k = self.normalize_k(k)
        try:
            result = tos_client.head_object(bucket_name, k)
            return True
        except tos.exceptions.TosServerError as err:
            if err.status_code == 404:
                return False
            else:
                traceback.print_exc()
                raise
        except:
            traceback.print_exc()
            raise


if __name__ == '__main__':
    tos_client = VolcTosClient()

    # print(tos_client.check_tos_file_exists('zhiyuexingchen/cn/podcast/xyz/00003/678dc708c3617f02a4ef1360.json'))

    data = tos_client.get_object('tos://video-data-downloader/zhiyuexingchen/cn/podcast/xyz/00003/678dc708c3617f02a4ef1360.m4a')
    print('len(data)', len(data))

    # ks = tos_client.list_prefix('tos://video-data-downloader/zhiyuexingchen/cn/podcast/xyz/00003/')
    # print('len(ks)', len(ks))
