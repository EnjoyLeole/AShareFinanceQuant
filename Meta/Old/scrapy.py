import re
import urllib
import urllib.request
import urllib.parse
import requests


def getHtml(url):
    page = response = requests.post(url)
    # html = page.read ()
    return page


def getImg(html):
    reg = r'src="(.+?\.jpg)" pic_ext'
    imgre = re.compile(reg)
    imglist = imgre.findall(html)
    x = 0
    for imgurl in imglist:
        urllib.urlretrieve(imgurl, '%s.jpg' % x)
        x = x + 1


M = 50
N = 50

v = [[0 for t in range(N)] for i in range(M)]

for i in range(M):
    for t in range(N - 1):
        x[i][t + 1] = x[i][t] + v[i][t]
        sum_a = v_jit = 0
        for j in range(M):
            sum_a = pow(abs(x[i][t] - x[j][t]), alpha)
            a[i][j] = stats.binom.rvs(n, p, size = 100) * 1 / (1 + sum_a)
            v_jit = v_jit + a[i][j] * (v[j][t] - v[i][t])
        v[i][t + 1] = v[i][t] + v_jit
#          plt.plot(t,v[i][t],'-')
# html = getHtml ("http://www.gravuregirlz.com")
# getImg (html)
