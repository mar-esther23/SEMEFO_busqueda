from re import sub as resub
from unidecode import unidecode
from statistics import mean
from warnings import warn

import pandas as pd
from numpy import nan


# Data structures

def flatten_list(regular_list):
    return [item for sublist in regular_list for item in sublist]


def simplify_string(text, str_case='lower'):
    """Simplify a string to remove special characters, double spaces and use lower case."""
    if type(text)==str:
        if str_case=='lower':
            text = text.lower()
        elif str_case=='upper':
            text = text.upper()
        else:
            text = text.capitalize()
        text = ' '.join([t for t in text.split()])
        text = unidecode(text)
        text = resub(r"[^a-zA-Z0-9]+", ' ', text)
        text = ' '.join([t for t in text.split()]) #paranoia
    return text


def replace_by_column(df, dic_replace):
    """Replace specific values by columns."""
    for col, replace in dic_replace.items():
        if col in df.columns:
            df[col] = df[col].replace(replace)
    return df


def groupby_column_values_and_separate_low_freq(df, target_col, min_rows=10, low_freq_label='low frequency'):
    high_freq = df[target_col].value_counts()
    high_freq = high_freq[high_freq>=min_rows].index
    df_groups = {i:df[df[target_col]==i] for i in high_freq}
    low_freq = df[~df[target_col].isin(high_freq)]
    if len(low_freq)>0:
        df_groups[low_freq_label] = low_freq
    return df_groups

def multicolumn_value_counts_with_filter(df, cols, sort_by=None, top_n=None, min_freq=None):
    if sort_by==None: sort_by = cols[0]
    if top_n==None: top_n = df.shape[0]
    if min_freq==None: min_freq = 1
    counts = df[cols].value_counts().head(top_n).reset_index() \
                     .sort_values(by=[sort_by,0], ascending=[True,False])
    counts = counts[counts[0]>=min_freq]
    counts = counts.reset_index(drop=True)
    counts.columns = cols + ['Frequency']
    return counts


# Manage dtypes

def failures_cast_to_type(i, dtype=int):
    """Tries to cast value af a type and returns the value if it fails."""
    try: dtype(i)
    except: return i
    return nan


def astype_and_separate_error(series, astype_function, suffix_success="_astype", suffix_error="_error", **kwargs):
    """
    Casts series astype and creates a column with succeses and an other with failures.
    
    Examples
    --------
    >>> astype_and_separate_error(df['Age'], pd.to_numeric, suffix_success='_int', downcast='integer')
    >>> astype_and_separate_error(df['Date'], pd.to_datetime, suffix_success='_date', format='%d/%m/%Y')
    """
    series_success = astype_function(series, errors='coerce', **kwargs)
    series_error = series.where(series_success.isna())
    df_astype = pd.concat([series_success,series_error],axis=1)
    df_astype.columns = [series.name+suffix_success, series.name+suffix_error]
    return df_astype

## String
def join_strings_with_nans(l, sep=' '):
    l = [s.strip() for s in l if type(s)==str]
    if len(l)>0:
        return sep.join(l)
    return nan

def punctuation_in_string(s, return_bool=False):
    from string import punctuation
    if type(s)==str:
        s = [i for i in s if i in set(punctuation)]
        s = ''.join(s)
        if return_bool:
            s = len(s)>0
        return s
    else: return nan

## Date

def expand_datetime(series_date):
    """Convert a pd.series to datetime and obtain year, month, day and weekday."""
    df_date = pd.concat([
                         series_date.dt.year,
                         series_date.dt.month,
                         series_date.dt.month_name(),
                         series_date.dt.day,
                         series_date.dt.day_name(),
                        ], axis=1 )
    names = ['year', 'month', 'monthname', 'day', 'dayname']
    df_date.columns = [series_date.name +'_' +n for n in names]
    return df_date


def translate_datetime(df):
    replace_date = {
                    'January':'Enero', 'February':'Febrero', 'March':'Marzo', 'April':'Abril', 
                    'May':'Mayo', 'June':'Junio', 'July':'Julio', 'August':'Agosto', 
                    'September':'Septiembre', 'Octuber':'Octubre', 'November':'Noviembre', 'December':'Diciembre', 
                    'Monday':'Lunes', 'Tuesday':'Martes', 'Wednesday':'Miércoles', 'Thursday':'Jueves', 
                    'Friday':'Viernes', 'Saturday':'Sábado', 'Sunday':'Domingo', 
                   }
    df = df.replace(replace_date)
    return df


## Int

def aproximate_int(s):
    """
    Aproximates an int from a float (dd.d) or a range (dd.dd-dd.dd) in string format.
    """
    if type(s)==str:
        try: 
            s = int(float(s))
        except ValueError:
            # try to solve range
            if '-' in s:
                s = s.split('-')
                try:
                    s = [float(i) for i in s]
                    s = int(mean(s))
                except: pass
    if type(s)==int: return s
    return nan

## Age

def divide_in_bins_for_age(series, step=10):
    """
    Special pd.cut for ages. 
    Includes special bin for age zero and groups all ages above 100 with last bin.
    """
    bins = [-1] + list(range(0, 100, step)) + [int(max(series)+1)]
    series_bins = pd.cut(series, bins)
    return series_bins


def replace_edades_menor_al_año(edad):
    if type(edad)==str:
        edad_ = simplify_string(edad)
        for s in ['dia', 'semana', 'mes']:
            if s in edad_: 
                n_edad = [i for i in edad_ if i.isdigit()]
                n_edad = float(''.join(n_edad))
                if 'dia' not in edad_ and n_edad>=12:
                    warn('Revisar edad {}, posiblemente mayor al año'.format(edad))
                return 0
    return edad


# Hierarchical

def list_to_nestdic(l, as_list=False):
    """
    Takes an iterable and returns a nested dic.
    If the iterable len is 1, returns the element.
    
    Examples
    --------
    >>> list_to_dic([1,2,3,4])
    {1: {2: {3: 4}}}
    >>> list_to_dic([1])
    1
    """
    dic = l.pop()
    if as_list: dic=[dic]
    while True:
        try: dic={l.pop():dic}
        except IndexError: break
    return dic

def create_hierarchical_catalog(df):
    # love
    from functools import reduce
    from operator import getitem
    # sort
    df.sort_values(df.columns.to_list())
    # initial data
    data = df.iloc[0].to_list()
    old  = df.iloc[0].to_list()
    data = list_to_nestdic(data)
    dic  = data
    # iterate
    for row in df.iloc[1:].iterrows():
        data = row[1].to_list()
        # compare and slice
        last_equal = [i==j for i,j in zip(data,old)].index(False) - 1
        if last_equal==-1:
            info = None
            poin = dic
        else: 
            info = reduce(getitem, data[0:last_equal], dic)
            poin = data[last_equal]
        data = list_to_nestdic(data[last_equal+1:])
        # create
        if info==None: dic.update(data)
        else: info[poin].update(data)
        # next
        old = row[1].to_list()
    return dic


# profiles
def profile_minimal(df, file_out, title=None, force=False, **kwargs):
    from os.path import exists
    from pandas_profiling import ProfileReport
    if force or not exists(file_out):
        if title==None: title=file_out
        print('Calculating profile',file_out)
        profile = ProfileReport(df, minimal=True, title=title, **kwargs)
        profile.to_file(file_out)
    else: print('Ignoring', file_out, 'file alredy exists')

def profile_by_categorical_column(df, file_out, target_col, min_rows=10, title=None, force=False, low_freq_label='low frequency', **kwargs):
    if title==None: title=file_out
    df_groups = groupby_column_values_and_separate_low_freq(df, target_col, min_rows, low_freq_label)
    for k, df_ in df_groups.items():
        file_out_ = file_out.replace('.html', '_{}_{}.html'.format(target_col, k))
        title_ = '{}/n{}: {}'.format(title, target_col, k)
        profile_minimal(df_, file_out_, title_, force, **kwargs)

def top_values_of_series_to_string(series, n_top=5, normalize=True, round_digits=4, sep=', '):
    top_values = series.value_counts(normalize=normalize, dropna=False).head(n_top)
    if normalize: top_values = top_values.round(round_digits)
    top_values = top_values.to_dict()
    text = ['{} ({})'.format(k,v) for k,v in top_values.items()]
    text = sep.join(text)
    return text










