# A Share Financial Quant / A股基本面量化


### Get Start 开始使用

* Fetch all data 下载所有数据

<pre><code>from Funds import *
Updater.everything()
</pre></code>

* Calculation update_stocks indicators 计算标的指标

<pre><code>update_stocks=Stocks('000678')
df=update_stocks.calc_list()
print(df)
</pre></code>

   

    


本项目参考了《基本面量化》(张然,2017 <http://www.gsm.pku.edu.cn/info/1195/19527.htm>)

武汉大学 廖珂 (Liao,K  Wuhan University)对本项目提供了细致的会计学专业支持

