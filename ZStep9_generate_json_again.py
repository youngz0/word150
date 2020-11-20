# coding=utf-8
from comnfunc import  *
import datetime
# import ods
t0 = datetime.datetime.now()
whichwd = '自'
con = convcsv2jsrd(whichwd)
genjsrules(whichwd)