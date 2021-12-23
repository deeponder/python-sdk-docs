import hashlib
import json
import logging
import os
import time
from io import BytesIO

import trafaret as t
import deepwisdom.errors as err

from deepwisdom._compat import Int, String
from .api_object import APIObject
from deepwisdom.enums import API_URL


def _get_upload_id(file_path):
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as f:
            data = f.read()
            return hashlib.md5(data).hexdigest() + "-" + str(time.time())
    else:
        logging.error("%s doesn't exist, no md5", file_path)
        return ""


_base_dataset_schema = t.Dict(
    {
        t.Key("dataset_id"): Int,
        t.Key("dataset_name", optional=True): String,
        t.Key("create_time"): String,
        t.Key("file_size"): Int,
        t.Key("file_type"): Int,
    }
)

class Dataset(APIObject):
    """

    """
    _converter = _base_dataset_schema.allow_extra("*")

    def __init__(
        self,
        dataset_id: int,
        dataset_name: str = None,
        create_time: str = None,
        file_size: int = None,
        file_type: int = None

    ):
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.create_time = create_time
        self.file_size = file_size
        self.file_type = file_type

    @classmethod
    def delete(cls, dataset_ids: list):
        """
        批量数据集
        Args:
            dataset_ids (list): 需要删除的数据集id列表

        Returns:

        """
        data = {
            "dataset_ids": dataset_ids
        }

        cls._client._delete(API_URL.DATASET_DELETE, data)

    @classmethod
    def dataset_search(cls, query: str):
        """
        查询数据集列表
        默认只拉一页，至多50条数据, 按更新的时间排序
        Args:
            query (str): 模糊查询关键字

        Returns:

        """
        data = {
            "query": query,
            "page": 1,
            "limit": 50,
            "sortrules": "-update_time"
        }
        server_data = cls._server_data(API_URL.DATASET_LIST, data)
        init_data = [dict(Dataset._safe_data(item)) for item in server_data]
        return [Dataset(**data) for data in init_data]

    @classmethod
    def create_from_file(
        cls,
        filename: str = None,
        model_type: int = None,
        annotation_type: int = 1,
        dataset_scene_id: int = 1,
        sep: str = "\\t",
        max_chunk_size: int = 10*1024*1024
    ):
        """
        从本地上传、创建数据集
        Args:
            filename: 本地数据集的绝对路径
            model_type: 模态类型。 0CSV,1VIDEO,2IMAGE,3SPEECH,4TEXT
            annotation_type: 标注类型。默认已标注1
            dataset_scene_id: 场景id。默认1 枚举
            sep: 表格分隔符。 枚举
            max_chunk_size: 分片上传的大小

        Returns:
            Dataset
        """

        dataset_id, msg = cls.dataset_upload(filename, model_type, annotation_type,
                                             dataset_scene_id, sep, max_chunk_size)
        if dataset_id < 0:
            logging.info(msg)
            raise err.UploadTrainDataError

        data = {
            "dataset_id": dataset_id
        }
        server_data = cls._server_data(API_URL.DATASET_INFO, data)

        return cls.from_server_data(server_data)

    @classmethod
    def create_from_dataset_id(cls, dataset_id):
        data = {
            "dataset_id": dataset_id
        }
        server_data = cls._server_data(API_URL.DATASET_INFO, data)

        return cls.from_server_data(server_data)

    @classmethod
    def _file_upload(cls, url, file, data):
        return cls._client._upload(url, data, file)

    @classmethod
    def dataset_upload(
        cls,
        file_path: str,
        model_type: int,
        annotation_type: int,
        dataset_scene_id: int,
        sep: str,
        max_chunk_size: int
    ):
        """
        数据集上传
        Args:
            file_path:
            model_type:
            annotation_type:
            dataset_scene_id:
            sep:
            max_chunk_size:

        Returns:

        """
        upload_id = _get_upload_id(file_path)
        if upload_id == "":
            return -1, file_path + " doesn't exist"
        # file_path = filepath.replace('\\', '/')
        file_names = os.path.split(file_path)
        filename = file_names[-1]
        # 上传准备
        prepare_data = {}
        prepare_data["modal_type"] = model_type
        prepare_data["filename"] = filename

        prepare_data["upload_id"] = upload_id

        resp = cls._client._post(API_URL.DATASET_PREPARE, prepare_data)

        if resp["data"]["ret"] != 1:
            return -1, resp

        _chunk_id_list = resp["data"]["chunk_id_list"]
        chunk_id_list = list()

        # 文件上传
        with open(file_path, 'rb') as fp:
            while 1:
                chunk = fp.read(max_chunk_size)
                if not chunk:
                    break
                chunk_id = hashlib.md5(chunk).hexdigest()
                chunk_id_list.append(chunk_id)
                if chunk_id in _chunk_id_list:
                    continue
                uploadData = {}
                uploadData["upload_id"] = upload_id
                uploadData["chunk_id"] = chunk_id
                chunk_fp = BytesIO(chunk)
                files = {
                    "file": (chunk_id, chunk_fp, 'application/octet-stream')
                }
                resp = cls._file_upload(API_URL.FILE_UPLOAD, files, uploadData)
                chunk_fp.close()
                if resp["data"]["ret"] != 1:
                    return -1, resp

        # 数据集上传
        dataset_deal_data = {}

        dataset_deal_data["filename"] = filename
        dataset_deal_data["upload_id"] = upload_id
        dataset_deal_data["chunk_id_list"] = json.dumps(chunk_id_list)
        dataset_deal_data["modal_type"] = model_type
        dataset_deal_data["sep"] = sep
        dataset_deal_data["annotation_type"] = annotation_type
        # 场景Id， 二分等
        dataset_deal_data["dataset_scene_id"] = dataset_scene_id

        resp = cls._client._post(API_URL.DATASET_UPLOAD, dataset_deal_data)

        if resp["data"]["ret"] != 1:
            return -1, resp

        # 3秒间隔，轮询获取
        # 获取数据集信息
        while True:
            dataset_query_data = {}
            dataset_query_data["upload_id"] = upload_id

            resp = cls._client._post(API_URL.DATASET_QUERY, dataset_query_data)
            logging.info(resp)
            if resp["errNo"] != 0 or resp["data"]["data_set_id"] == -1:
                return -1, resp

            # id不为0生成完成
            if resp["data"]["data_set_id"] > 0:
                return resp["data"]["data_set_id"], "success"

            time.sleep(3)

    @classmethod
    def modify_dataset(cls, dataset_id: int, new_name: str = ""):
        """
        重命名数据集
        Args:
            dataset_id (int): 数据集id
            new_name (str): 新的名字

        Returns:

        """

        data = {
            "dataset_id": dataset_id,
            "file_name": new_name
        }

        cls._client._patch(API_URL.DATASET_MODIFY, data)


class PredictDataset(APIObject):
    _converter = t.Dict(
        {
            t.Key("dataset_id"): Int,
            t.Key("dataset_name", optional=True): String
        }
    ).allow_extra("*")

    def __init__(
            self,
            project_id: int,
            dataset_id: int,
            dataset_name: str = None
    ):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name


