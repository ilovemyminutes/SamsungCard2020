import os, glob
import numpy as np
import pandas as pd

data_path = '../raw/'
trend_w_demo = pd.read_csv(data_path + '[Track2_데이터1] trend_w_demo.csv', engine='python')

# # 분석 개요
# 
# - 목표
#     - SC에서는 무엇을 원하는가? : 팬데믹 시대를 극복할 마케팅 전략 수립
#         - 전년 대비 올해, 어떤 업종의 구매량이 어떤 고객군에서 증가 or 감소했는가?
#         - 그 중에서도 어떤 업종이 올해 두 달간 증감폭이 큰가?
#             - ex) 취미 업종의 구매량이 대학생 자녀를 둔 40대 남성에서 크게 증가했다. 특히, 월별 증가량 추이가 두드러진다.
#             - => 고객군 타겟 마케팅 전략 수립

# In[ ]:


train = trend_w_demo.copy()
train = train.rename(columns=dict(성별구분='Sex', 연령대='Age', 기혼스코어='Marriage', 유아자녀스코어='Youth', 초등학생자녀스코어='Elementary',
                                  중고생자녀스코어='Mid-High', 대학생자녀스코어='University', 전업주부스코어='Housewife'))

tot_var_list = train.columns[2:].tolist()
print(tot_var_list)


# # 고객군별 전체 구매량

# In[ ]:


def cnt_df_check(df_, var_list):
    data = df_.copy()
    output = data.groupby(var_list)['YM'].count().to_frame('Value').reset_index()
    return output

def cnt_tf_series(series_):
    data = series_.copy()
    srch_list = data.drop(index=['Value'])
    temp = []
    for i in range(len(srch_list)):
        temp.append(f'{str(srch_list.index[i])}={str(srch_list[i])}')
    output = '|'.join(temp)
    return output

def cnt_tf_df(df_):
    data = df_.copy()
    data['Segment'] = data.apply(lambda x: cnt_tf_series(x), axis=1)
    output = data[['Segment', 'Value']]
    return output

def cnt_funnel(df_, var_list):
    from itertools import combinations
    data = df_.copy()

    output = pd.DataFrame()
    for i in range(len(var_list)):
        list_ = list(combinations(var_list, i + 1))
        for j in list_:
            temp = cnt_df_check(data, list(j))
            temp = cnt_tf_df(temp)
            output = pd.concat([output, temp], axis=0, ignore_index=True)
    return output


# In[ ]:


cnt_table = cnt_funnel(train, tot_var_list)

# In[ ]:


print('중앙값:', cnt_table['Value'].median())
valid_seg = cnt_table[cnt_table['Value'] >= 735]['Segment'].tolist()
print('유효한 고객군의 수:', len(valid_seg))


# # 업종별 고객군별 증가량/증가율

# In[ ]:


def yoy_compare(df_, var_list):
    data = df_.copy()
    data['Year'] = data['YM'].apply(lambda x: 2020 if '2020' in str(x) else 2019)
    temp = data.groupby(['Year', 'Category'] + var_list)['YM'].count().to_frame('Value')
    output = round((temp.loc[2020] - temp.loc[2019]) / temp.loc[2019] * 100, 2).reset_index()
    return output


def yoy_tf_series(series_):
    data = series_.copy()
    srch_list = data.drop(index=['Category', 'Value'])

    var_list = []
    for i in range(len(srch_list)):
        var_list.append(f'{str(srch_list.index[i])}={str(srch_list[i])}')

    output = '|'.join(var_list)
    return output


def yoy_tf_df(_df, censor_list):
    data = _df.copy()
    data['Segment'] = data.apply(lambda x: yoy_tf_series(x), axis=1)
    output = data[['Category', 'Segment', 'Value']]
    return output


def rate_funnel(df_, var_list, censor_list):
    from itertools import combinations
    data = df_.copy()

    df = pd.DataFrame()
    for i in range(len(var_list)):
        list_ = list(combinations(var_list, i + 1))
        for j in list_:
            print(j)
            temp = yoy_compare(data, list(j))
            temp = yoy_tf_df(temp, censor_list)
            df = pd.concat([df, temp], axis=0, ignore_index=True)

    df = df[df['Value'].notnull()]
    df = df[df['Segment'].isin(censor_list)]
    df = df.reset_index(drop=True)

    output = df.copy()
    return output


# In[ ]:


rate_table = rate_funnel(train, tot_var_list, valid_seg)

# In[ ]:


rate_table.to_pickle('./data/yoy_rate_table.pkl')
