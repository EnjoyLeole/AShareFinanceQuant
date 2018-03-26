from simpledbf import Dbf5
import dbf


def dbf2df(path):
    db = Dbf5(path)
    df = db.to_dataframe()
    return df


def print_dbf(path):
    # path = 'd:\order_updates.dbf'
    db = dbf.Table(path)
    db.open()
    print(dbf.field_names)
    for record in db:
        print(record)
