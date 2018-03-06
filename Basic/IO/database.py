import datetime
import numpy as np
import pandas as pd
import pandas.io.pytables as pt
import sqlalchemy as sql
from pymongo import MongoClient


class DataHandler(object):
    table_list = None

    def __init__(self, api, user, password, host, db, port, add = ""):
        self.schema = db
        self.Engine = sql.create_engine("%s://%s:%s@%s:%s/%s" % (api, user, password, host, port, db) + add,
                                        pool_recycle = 5)

    @staticmethod
    def filename_store(series = ''):

        store_path = 'D:/Program Files/StockData/stock_data%s.h5'
        return store_path % series

    def _get_table_attr_sql(self, table_name, schema = None):
        raise NotImplementedError()

    def _get_table_attr_value_type(self, table_name, schema = None):
        raise NotImplementedError()

    def _match_values(self, if_update):
        def _match(table_name, row):
            cols = self.get_table_attr(table_name)
            # print(cols)
            values = ""
            for col_name in cols:
                # print(col_name)
                if values != "":
                    values += ","
                if if_update:
                    values += self.prefix_name(col_name) + "="

                if hasattr(row, col_name):
                    val = getattr(row, col_name)
                    if val is None:
                        values += "Default"
                    else:
                        # print(col_name,type(val),val)
                        if type(val) is str:
                            values += "'%s'" % val
                        elif type(val) is np.datetime64:
                            if pd.isnull(val):
                                val = datetime.datetime.now()
                            values += "'%s'" % val
                        elif type(val) == pd.tslib.Timestamp:
                            values += "'%s'" % val
                        # elif type(val)==pd.Series:
                        #     values+=str(val[0])
                        elif np.isnan(val):
                            values += "Default"
                        else:
                            values += str(val)
                else:
                    values += "Default"
            return values

        # if_update = if_update
        return _match

    def table_contain(self, table_name):
        for x in self.table_list:
            if x == table_name:
                return True
        return False

    def table_drop(self, table_name):
        if self.table_contain(table_name):
            self.execute("DROP TABLE  IF EXISTS   %s" % self.prefix_name(table_name.lower()))
            self.table_list.remove(table_name)

    def table_read(self, table_name, chunksize = None):
        table_name = table_name.lower()
        return pd.read_sql_table(table_name, self.Engine, parse_dates = True, chunksize = chunksize)

    def table_save(self, df, table_name, append = True, schema = None, chunksize = None, index_label = None):
        table_name = table_name.lower()
        if append is False:
            try:
                self.execute("delete from %s" % self.prefix_name(table_name))
            except Exception as e:
                print(e)
                pass
        if_exists = 'append'
        df.to_sql(table_name, self.Engine, if_exists = if_exists, index = False, chunksize = chunksize,
                  index_label = index_label)
        # if index_label is not None:
        #     try:
        #         # drop_sql = "alter table %s drop primary key" % self.prefix_name(table_name)
        #         # self.execute(drop_sql)
        #         sql_cmd = 'ALTER TABLE %s ADD PRIMARY KEY (%s)' % (
        #             self.prefix_name(table_name), self.prefix_name(index_label))
        #         self.execute(sql_cmd)
        #     finally:
        #         pass

    def sql_read(self, sql_cmd, params = None, chunksize = None):
        return pd.read_sql_query(sql_cmd, self.Engine, params = params, parse_dates = True, chunksize = chunksize)

    def execute(self, sql_cmd):
        with self.Engine.connect() as conn:
            return conn.execute(sql_cmd)

    def get_table_attr(self, table_name, schema = None):
        table_name = table_name.lower()
        sql_cmd = self._get_table_attr_sql(table_name, schema)
        result = self.execute(sql_cmd)
        attr = []
        for row in result:
            if row[0][0] != "." and not attr.__contains__(row[0]):
                attr.append(row[0])
        return attr

    def print_net_class(self, table_name, schema = None):
        # db_type_to_net={"float":"Double",
        #                 "double":"Double",
        #                 "int":"Int",
        #                 "bigint":"Int",
        #                 "varchar":"String",}


        table_name = table_name.lower()
        sql = self._get_table_attr_value_type(table_name, schema)
        # print(sql)
        result = self.execute(sql)
        class_def = "class %s { " % table_name
        for row in result:
            declareType = "string" if row[1] == "varchar" else row[1]
            class_def += "public %s %s {get;set;}\n" % (declareType, row[0])
        class_def += "}"
        return class_def

    def prefix_name(self, name):
        return name

    def insert(self, table_name, obj):
        table_name = table_name.lower()
        match = self._match_values(False)
        values = match(table_name, obj)
        sql_cmd = "insert into %s values (%s)" % (self.prefix_name(table_name), values)
        self.execute(sql_cmd)

    def update(self, table_name, obj, primary_key, key_value = None):
        table_name = table_name.lower()
        key_value = getattr(obj, primary_key) if key_value is None else key_value
        match = self._match_values(True)
        values = match(table_name, obj)
        sql_cmd = "update  %s set %s where %s = %s" % (self.prefix_name(table_name), values, primary_key, key_value)
        self.execute(sql_cmd)

    def insert_df(self, table_name, df):
        table_name = table_name.lower()
        for row in df.iterrows():
            row = row[1]
            self.insert(table_name, row)

    def update_df(self, table_name, df, primary_key):
        table_name = table_name.lower()
        for row in df.iterrows():
            row = row[1]
            self.update(table_name, row, primary_key)


class MySqlHandler(DataHandler):


    def __init__(self, db, user = "root", password = "temppd", host = "localhost", port = 3306):
        super().__init__("mysql+pymysql", user, password, host, db, port, "?charset=utf8")

        self.table_list = \
            pd.read_sql_query("select table_name from information_schema.tables where table_schema='%s'" % db,
                              self.Engine)['table_name'].tolist()

    def _get_table_attr_sql(self, table_name, schema = None):
        if schema == None:
            schema = self.schema
        sql_cmd = "select COLUMN_NAME from information_schema.COLUMNS " \
                  "where table_name = '%s' and table_schema='%s'" % (table_name, schema)
        return sql_cmd

    def _get_table_attr_value_type(self, table_name, schema = None):
        if schema == None:
            schema = self.schema
        return "select COLUMN_NAME,DATA_TYPE from information_schema.COLUMNS " \
               "where table_name = '%s' and table_schema='%s'" % (table_name, schema)


class PgSQLHandler(DataHandler):
    @classmethod
    def at106(cls):
        return cls("cad", host = "10.1.1.6")

    def __init__(self, db, user = "postgres", password = "melody", host = "localhost", port = 5432):
        super().__init__("postgresql+psycopg2", user, password, host, port, db)
        self.table_list = pd.read_sql_query("SELECT tablename FROM pg_tables", self.Engine)['tablename'].tolist()

    def _get_table_attr_sql(self, table_name):
        sql_cmd = "select COLUMN_NAME from information_schema.COLUMNS " \
                  "where table_name = '%s'" % table_name
        return sql_cmd

    def prefix_name(self, name):
        return "\"%s\"" % name


def hdf2pgsql(self, hdf_handler):
    for ds_id in hdf_handler.table_list:
        df = hdf_handler.table_read(ds_id)
        self.table_save(df, ds_id, append = False)
        print(ds_id)


class HDFHandler(DataHandler):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.table_list = self._hdf_ds_list()

    def _hdf_ds_list(self):
        """Backup all tables"""

        my_file = pt._tables().open_file(self.filename, mode = 'a')

        temp = []
        for node in my_file.list_nodes('/'):
            temp.append(node._v_name)
        # bk = pd.read_hdf(filename_store('_old'), bkid)
        # bk = bk.sort_index()
        # bk.to_hdf(filename_store('b'), bkid, append=False, format='t')
        my_file.close_cad()
        return temp

    def table_contain(self, dataset_id):
        for x in self.table_list:
            if x == dataset_id:
                return True
        return False

    def table_read(self, table_name, chunksize = None):
        return pd.read_hdf(self.filename, table_name)

    def table_save(self, df, table_name, file_no = "", append = True, format = 't', schema = None):
        df.to_hdf(self.filename + file_no, table_name, append = append, format = format)

    def backup_hdf(self):
        """Backup all tables"""
        for ds_name in self.table_list:
            ds = self.table_read(ds_name)
            self.table_save(ds, ds_name, file_no = "backup")


class MongoHandler(DataHandler):
    def __init__(self, db):
        super().__init__()
        self.Mongo_Stock = MongoClient('localhost', port = 27017)[db]

    def table_save(self, df, table_name, append = True, schema = None, chunksize = None):
        dict = df.to_dict('records')
        self.Mongo_Stock[table_name].insert_many(dict)

    def table_read(self, table_name, chunksize = None):
        cursor = self.Mongo_Stock[table_name].find()
        df = pd.DataFrame(list(cursor))
        return df

    def table_contain(self, dataset_id):
        pass
