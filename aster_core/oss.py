import oss2
import os
from aster_core.token_config import accessKeyId, accessKeySecret, current_directory
from concurrent.futures import ThreadPoolExecutor, as_completed

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
            oss2.resumable_download(bucket, url, out_file, num_threads=4)
        else:
            cmd = f'{current_directory}/ossutil cp oss://{bucket_name}/{url} {out_file} --config-file {current_directory}/.ossutil_config > /dev/null 2>&1'
            os.system(cmd)
    else:
        # print(f'Already download {out_file}, skip')
        pass
        
    return out_file

def upload_file_to_oss(url, in_file, bucket_name='geocloud'):
    if os.path.exists(in_file):
        # 配置访问凭证
        auth = oss2.Auth(accessKeyId, accessKeySecret)

        # Endpoint和Region
        endpoint = 'oss-cn-hangzhou-zjy-d01-a.ops.cloud.zhejianglab.com/'
        region = 'cn-hangzhou'


        bucket = oss2.Bucket(auth, endpoint, bucket_name, region=region)
        oss2.resumable_upload(bucket, url, in_file, num_threads=4)

    else:
        # print(f'Already download {out_file}, skip')
        pass
        
    return in_file