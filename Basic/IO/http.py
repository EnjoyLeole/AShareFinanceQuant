# -*- coding: utf-8 -*-
import http.client
import traceback
import urllib

import gzip
from io import BytesIO

HTTP_OK = 200
HTTP_AUTHORIZATION_ERROR = 401


class MyClient:
    domain = ""
    # port = 443
    # token = ''
    # 设置因网络连接，重连的次数
    reconnectTimes = 2
    httpClient = None

    @classmethod
    def _connect(cls):
        cls.httpClient = http.client.HTTPSConnection(cls.domain, 80, timeout = 60)

    # def __init__(self):
    #     self._connect()
    #
    # def __del__(self):
    #     if self.httpClient is not None:
    #         self.httpClient.close()
    @classmethod
    def getData(cls, path):
        result = None
        if cls.httpClient is None:
            cls._connect()
        for i in range(cls.reconnectTimes):
            try:
                # set http header here
                cls.httpClient.request('GET', path)
                # make request
                response = cls.httpClient.getresponse()
                result = response.read()
                # compressedstream = BytesIO(result)
                # gziper = gzip.GzipFile(fileobj = compressedstream)
                # try:
                #     result = gziper.read()
                # except:
                #     pass
                return response.status, result
            except Exception as e:
                if i == cls.reconnectTimes - 1:
                    raise e
                if cls.httpClient is not None:
                    cls.httpClient.close()
                cls._connect()
        return -1, result
