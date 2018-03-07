import pandas as pd
import urllib
import urllib.request
import urllib.parse
import requests
import datetime


class MyGetter(object):
    request_url = "http://www.cninfo.com.cn/cninfo-new/disclosure/szse_latest"
    pdfheader = "http://www.cninfo.com.cn/cninfo-new/disclosure/szse/bulletin_detail/true/"

    headers = {
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Accept-Encoding': 'gzip, deflate',
        'Accept-Language': 'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3',
        'Content-Length': '230',
        'Content-Type	': 'application/x-www-form-urlencoded; charset=UTF-8',
        'Host': 'www.cninfo.com.cn',
        'Referer': 'http://www.cninfo.com.cn/cninfo-new/disclosure/szse',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:47.0) Gecko/20100101 Firefox/47.0',
        'X-Requested-With': 'XMLHttpRequest'
    }

    def postdata(self, page):
        return urllib.parse.urlencode(
            {'stock': '',
             'searchkey': '',
             'plate': '',
             'category': '',
             'trade': '',
             'column': 'szse',
             'columnTitle': '深市公告',
             'pageNum': '%s' % page,
             'pageSize': '50',
             'tabName': 'latest',
             'sortName': '',
             'sortType': '',
             'limit': '',
             'showTitle': '',
             'seDate': '2016-06-17'})

    def __init__(self):
        self.annolist = []

    def post(self):
        try:
            i = 1
            while (True):
                response = requests.post(self.request_url, data=self.postdata(i), headers=self.headers)
                size = self._add_list(response)
                i += 1
                if size == 0:
                    break
        except:
            print(i)

    def _add_list(self, response):
        content = response.json()
        # typelist=content["categoryList"]
        announcements = content["classifiedAnnouncements"]
        size = len(announcements)
        print("公告企业数" + size.__str__())
        for idx, anno in enumerate(announcements):
            for i in range(0, len(anno)):
                # print(anno)
                if (i == 0 and idx == 0) or (i == len(anno) - 1 and idx == size - 1):
                    print(anno[i]['secName'] + '--' + anno[i]['announcementTitle'])
                    # print(anno[i])
                self.annolist.append(anno[i])
        return size

    def print(self):
        print("总输出公告数" + self.annolist.__len__().__str__())
        df = pd.DataFrame(self.annolist)
        df.reset_index()

        csv = df.ix[:, ['secName', 'announcementTitle', 'announcementTypeName', 'important']]
        csv['secCode'] = df.apply(lambda x: 'sz' + x['secCode'].__str__().zfill(6), axis=1)
        csv['Date'] = df.apply(lambda x: datetime.datetime.fromtimestamp(float(x['announcementTime']) / 1000), axis=1)
        csv['pdfurl'] = self.pdfheader + df.ix[:, 'announcementId']
        csv.to_csv("D:/test.csv", mode='w')


if __name__=="main":
    getter = MyGetter()
    getter.post()
    getter.print()

