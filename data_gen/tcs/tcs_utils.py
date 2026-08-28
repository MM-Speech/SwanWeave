import os
import traceback

os.environ['CONSUL_HTTP_HOST'] = '10.37.4.197'
os.environ['CONSUL_HTTP_PORT'] = '2280'

from time import time
import hashlib
import uuid
import requests

import json

def get_headers():
    """
    获取headers
    :return:
    """
    # 分配的access_key, access_secret
    access_key = 'UM63C1THT9'
    access_secret = 'PSY2JVH4H7UB5D2WBW5GKSSCJR9FX0TLST9UGDE039H7YDNR64'

    timestamp = str(int(time()))

    nonce = key = uuid.uuid4().hex
    _list = [access_secret, timestamp, nonce]
    _list.sort()
    signature = hashlib.sha1(''.join(_list).encode('utf-8')).hexdigest()
    headers = {
        'X-AccessKey': access_key,
        'X-Signature': signature,
        'X-Timestamp': timestamp,
        'X-Nonce': nonce
    }
    return headers


def upload_single_task(project_id, object_id, params):
    """
    根据project_id，object_id上传单条任务
    :param project_id: string
    :param object_id: string
    :param params: dict
    :return:
    """
    r = requests.post('https://tcs.bytedance.net/api/v2/create_task/',
                      data=dict(project_id=project_id, object_id=object_id,
                                object_data=json.dumps(params)),
                      headers=get_headers(), verify=False)
    ret = eval(r.text)
    if 'message' in ret and ret['message'] == 'success':
        return {'task_id': ret['data']['task_id'], "object_id": object_id, "success": "1"}
    else:
        return {"object_id": object_id, "success": "1"}
