from .formulary import *


class StockGroups:

    def __init__(self, code_list):
        self.code_list = code_list

    def all_targets(self):
        def  calc(code):
            print(code + ' start')
            stk=Stocks(code)
            stk.all_target()
            print(code+' saved')
        DMgr.iter_stocks(calc,'target_calc')
