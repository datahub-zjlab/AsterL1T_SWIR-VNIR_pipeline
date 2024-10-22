import oss2
from oss2.exceptions import NoSuchKey
import os
from aster_core.token_config import accessKeyId, accessKeySecret, current_directory

def get_bucket(bucket_name):
    auth = oss2.Auth(accessKeyId, accessKeySecret)
    # Endpoint和Region
    endpoint = 'oss-cn-hangzhou-zjy-d01-a.ops.cloud.zhejianglab.com/'
    region = 'cn-hangzhou'
    bucket = oss2.Bucket(auth, endpoint, bucket_name, region=region)
    return bucket

def download_file_from_oss(url, bucket_name='geocloud', out_file='./tmp.hdf',
                           overwrite=False, oss_util_flag=False):
    if not os.path.exists(out_file) or overwrite:
        # 配置访问凭证
        auth = oss2.Auth(accessKeyId, accessKeySecret)

        # Endpoint和Region
        endpoint = 'oss-cn-hangzhou-zjy-d01-a.ops.cloud.zhejianglab.com/'
        region = 'cn-hangzhou'

        if not oss_util_flag:
            # Bucket信息
            bucket = oss2.Bucket(auth, endpoint, bucket_name, region=region)
            try:
                oss2.resumable_download(bucket, url, out_file, num_threads=4)
            except NoSuchKey:
                print(f"File '{url}' does not exist in OSS.")
                return None
            except Exception as e:
                print(f"An error occurred while downloading the file {url}: {e}")
                return None
        else:
            cmd = f'{current_directory}/ossutil cp oss://{bucket_name}/{url} {out_file} --config-file {current_directory}/.ossutil_config > /dev/null 2>&1'
            os.system(cmd)
    else:
        # print(f'Already download {out_file}, skip')
        pass
        
    return out_file

import contextlib
import io

def suppress_print(func):
    def wrapper(*args, **kwargs):
        # 使用 io.StringIO 捕获标准输出
        with contextlib.redirect_stdout(io.StringIO()):
            result = func(*args, **kwargs)
        return result
    return wrapper

@suppress_print
def upload_file_to_oss(url, in_file, bucket_name='geocloud',print_flag=False):
    if os.path.exists(in_file):
        # 配置访问凭证
        auth = oss2.Auth(accessKeyId, accessKeySecret)

        # Endpoint和Region
        endpoint = 'oss-cn-hangzhou-zjy-d01-a.ops.cloud.zhejianglab.com/'
        region = 'cn-hangzhou'

        bucket = oss2.Bucket(auth, endpoint, bucket_name, region=region)

        # 检查OSS上是否已经存在该文件
        if not bucket.object_exists(url):
            oss2.resumable_upload(bucket, url, in_file, num_threads=2)
        else:
            if print_flag:
                print(f'File {url} already exists in OSS, skipping upload.')
    else:
        if print_flag:
            print(f'File {in_file} does not exist locally, skipping upload.')
        
    return in_file
