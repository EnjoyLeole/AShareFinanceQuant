#A Share Financial Quant / A股基本面量化

###主要功能


* 免费数据获取（webio.py）

    基于网易数据及tushare，可自动更新
    * 股票历史tick及公司财务季报
    * 指数tick
    * 主要宏观数据

* 数据管理及清洗（dataio.py）
    * 基于csv文件系统的对应数据存取及遍历、合并功能
    * 中文字段名转化为英文标准名称
    * 不同表中同名列的自动化比较和选择
    * 对财务报表做ttm处理
    

* 公式化的财务指标计算(formulary.py)
    * 对股票和指数进行财务指标计算
    * 基于csv表格的公式定义
    * 支持对数据进行ttm、期内平均等前处理
    * 支持对指标进行几何平均、倾斜平均、方差等计算

    
### 部分已实现的财务指标

* __MG & MS__ 
    盈利能力与稳定性
* __M-Score__ 
    >判断利润操纵的经典方法  
    <https://en.wikipedia.org/wiki/Beneish_M-Score>
* __Z-Score__ 
    >判断破产风险的经典方法   
    <https://en.wikipedia.org/wiki/Altman_Z-score>
* __CHS Model__ 
    >基于时间序列的破产风险判断    
    In Search of Distress Risk 
    <https://scholar.harvard.edu/campbell/publications/search-distress-risk>
* __CI__    过度投资比例

---