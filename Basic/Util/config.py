from Basic.Util.classext import *
from Basic.IO.database import *


class Config(object):
    _db = None
    _db_name = 'cad'
    project_table = "ConfigProject"
    upper_flag = 1

    @classproperty
    def db_handler(self):
        if self._db is None:
            self._db = MySqlHandler(self._db_name)
        return self._db

    @classmethod
    def set_db_handler(cls, db_handler, flag=1):
        cls._db = db_handler
        cls.upper_flag = flag

    @classmethod
    def get_project_candidates(cls):
        select = "select pid from %s where %s is NULL or %s <%s" % (
            cls.db_handler.prefix_name(cls.project_table),
            cls.db_handler.prefix_name("ifEmailed"),
            cls.db_handler.prefix_name("ifEmailed"),
            cls.upper_flag)
        pid_candidates = cls.db_handler.sql_read(select)
        return pid_candidates

    @classmethod
    def update_project(cls, pid):
        cls.db_handler.execute(
            "UPDATE \"ConfigProject\" SET \"ifEmailed\"=1, \"sendtime\"='%s' WHERE pid='%d'"
            % (datetime.datetime.now(), pid))

    @classmethod
    def setup_config(cls, db_handler):
        local = cls.db_handler
        for t in local.table_list:
            if t.count("config") and t != "configproject":
                df = cls.table_read(t)
                db_handler.table_save(df, t, append=False)

    @classmethod
    def table_read(cls, dataset_id):
        return cls.db_handler.table_read(dataset_id)

    @classmethod
    def table_save(cls, df, dataset_id, append=True, schema=None):
        cls.db_handler.table_save(df, dataset_id, append, schema)

    @classmethod
    def update(cls, glass, primary_key, table_name=""):
        table_name = (glass.__name__ if table_name == "" else table_name).lower()
        cls.db_handler.update(table_name, glass, primary_key)

    @classmethod
    def load_setting(cls, glass, table_id, value, table_name=""):
        table_name = (glass.__name__ if table_name == "" else table_name).lower()
        df = cls.db_handler.table_read(table_name)
        df = df[df[table_id] == value]
        for col in df:
            setattr(glass, col, df[col].values[0])

    @classmethod
    def backup(cls, glass):
        df = cls.db_handler.table_read(glass.__name__.lower())
        df.to_csv("%s.csv" % glass.__name__.lower())
        return df

    @classmethod
    def restore(cls,glass):
        df = pd.read_csv("%s.csv" % glass.__name__.lower())
        cls.db_handler.dataset_save(df, glass.__name__.lower())

    @classmethod
    def concat(cls,cls_list):
        dfs = []
        for glass in cls_list:
            temp = cls._get_setting(glass, if_wide=False)
            dfs.append(temp)
        df = pd.concat(dfs, axis=1)
        return df

    @staticmethod
    def _get_setting(glass, if_wide=False):
        df = pd.DataFrame()
        for field in glass.__dict__:
            value = getattr(glass, field)
            if if_wide:
                if not callable(value) and not isinstance(value, object) and field[0] != '_':
                    df[field] = [value]
            else:
                if (isinstance(value, (int, float, str, classproperty, np.int64)) or value is None) and field[0] != '_':
                    df[field] = [value]

        df = df.sort_index(axis=1)
        return df
