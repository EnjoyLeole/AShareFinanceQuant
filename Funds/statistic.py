from .formulary import *


class StockGroups:

    def __init__(self, code_list):
        self.code_list = code_list

    def all_targets(self):
        for code in self.code_list:
            print(code + ' start')
            stk=Stocks(code)
            stk.all_target()
            print(code+' saved')