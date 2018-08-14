import sqlite3
from sqlite3 import Error


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
    return None

def add_field(conn, create_table_sql, column_name):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute("alter table %s add column '%s' 'float'" % (create_table_sql, column_name))
    except Error as e:
        print(e)


def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)

def insert_news(conn, new):
    """
    Create a new new into the news table
    :param conn:
    :param new:
    :return: new id
    """
    sql = ''' INSERT INTO BBBNews(authors,keywords,publish_date,summary,text,title,top_image,url,score,newspaper)
              VALUES(?,?,?,?,?,?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, new)
    return cur.lastrowid

def update_news(conn, new):
    """
    update priority, begin_date, and end date of a task
    :param conn:
    :param new:
    :return: new id
    """
    sql = ''' UPDATE BBBNews
              SET url = ? ,
                  title = ? ,
                  authors = ?,
                  publish_date = ?,
                  text = ?
              WHERE id = ?'''
    cur = conn.cursor()
    cur.execute(sql, new)

def delete_news(conn, id):
    """
    Delete a task by task id
    :param conn:  Connection to the SQLite database
    :param id: id of the task
    :return:
    """
    sql = 'DELETE FROM BBBNews WHERE id=?'
    cur = conn.cursor()
    cur.execute(sql, (id,))

def delete_all_news(conn):
    """
    Delete all rows in the tasks table
    :param conn: Connection to the SQLite database
    :return:
    """
    sql = 'DELETE FROM BBBNews'
    cur = conn.cursor()
    cur.execute(sql)

def select_all_tasks(conn):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM BBBNews")

    rows = cur.fetchall()

    for row in rows:
        print(row)

def check_if_exists(conn, url):
    cur = conn.cursor()
    sql = 'SELECT url FROM BBBNews WHERE url = ?'
    cur.execute(sql, (url,))
    data=cur.fetchone()
    if data is None:
        return 0
    else:
        return 1
