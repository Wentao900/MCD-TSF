import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.timefeatures import time_features
import warnings
from utils.prepare4llm import get_desc
from utils.rag_cot import ScaleAwareRAGCoT

warnings.filterwarnings('ignore')

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 text_len=1, scaler_type='standard',
                 use_scale_rag_cot=False, cot_model_name='gpt2-medium',
                 cot_local_files_only=True, cot_max_new_tokens=64,
                 rag_stage1_topk=12, rag_stage2_topk=3,
                 use_longformer_rerank=False, longformer_model_name='allenai/longformer-base-4096',
                 longformer_local_files_only=True, longformer_max_length=2048,
                 rag_long_topn=24, rag_cache_guidance=True):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[-1]
        # init
        assert flag in ['train', 'test', 'valid']
        type_map = {'train': 0, 'valid': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.text_len = text_len
        self.use_scale_rag_cot = use_scale_rag_cot
        self.cot_model_name = cot_model_name
        self.cot_local_files_only = cot_local_files_only
        self.cot_max_new_tokens = cot_max_new_tokens
        self.rag_stage1_topk = rag_stage1_topk
        self.rag_stage2_topk = rag_stage2_topk
        self.use_longformer_rerank = use_longformer_rerank
        self.longformer_model_name = longformer_model_name
        self.longformer_local_files_only = longformer_local_files_only
        self.longformer_max_length = longformer_max_length
        self.rag_long_topn = rag_long_topn
        self.rag_cache_guidance = rag_cache_guidance
        self.guidance_cache = {}

        self.root_path = root_path
        self.data_path = data_path
        self.data_prefix = data_path.split('.')[0]
        
        if scale:
            if scaler_type == 'minmax':
                self.scaler = MinMaxScaler(feature_range=(-1, 1))
            elif scaler_type == 'standard':
                self.scaler = StandardScaler()
            else:
                scaler_type = 'minmax'
                self.scaler = MinMaxScaler(feature_range=(-1, 1))
        else:
            self.scaler = None
        self.scaler_type = scaler_type
        self.__read_data__()
        self.domain = data_path.split('/')[0]
        self.desc = get_desc(self.domain, self.seq_len, self.pred_len)
        self.rag_cot = None
        if self.use_scale_rag_cot:
            self.rag_cot = ScaleAwareRAGCoT(
                self.txt_search,
                rag_stage1_topk=self.rag_stage1_topk,
                rag_stage2_topk=self.rag_stage2_topk,
                cot_model_name=self.cot_model_name,
                cot_local_files_only=self.cot_local_files_only,
                cot_max_new_tokens=self.cot_max_new_tokens,
                use_longformer_rerank=self.use_longformer_rerank,
                longformer_model_name=self.longformer_model_name,
                longformer_local_files_only=self.longformer_local_files_only,
                longformer_max_length=self.longformer_max_length,
                rag_long_topn=self.rag_long_topn,
            )
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1
        

    def __read_data__(self):
        df_num = pd.read_csv(os.path.join(self.root_path, 'numerical', self.data_path))
        df_report = pd.read_csv(os.path.join(self.root_path, 'textual', self.data_prefix + '_report.csv'))
        df_search = pd.read_csv(os.path.join(self.root_path, 'textual', self.data_prefix + '_search.csv'))

        df_num = df_num.dropna(axis='index', how='any', subset=['OT'])
        df_report = df_report.dropna(axis='index', how='any', subset=['fact'])
        df_search = df_search.dropna(axis='index', how='any', subset=['fact'])

        df_num['date'], df_num['start_date'], df_num['end_date'] = pd.to_datetime(df_num['date']), pd.to_datetime(df_num['start_date']), pd.to_datetime(df_num['end_date'])
        df_report['start_date'], df_report['end_date'] = pd.to_datetime(df_report['start_date']), pd.to_datetime(df_report['end_date'])
        if 'start_date' in df_search.columns and 'end_date' in df_search.columns:
            df_search['start_date'] = pd.to_datetime(df_search['start_date'], errors='coerce')
            df_search['end_date'] = pd.to_datetime(df_search['end_date'], errors='coerce')
            df_search = df_search.dropna(axis='index', how='any', subset=['start_date', 'end_date'])
        else:
            df_search = df_search.iloc[0:0].copy()

        df_num = df_num.sort_values('date', ascending=True).reset_index(drop=True)
        df_report = df_report.sort_values('start_date', ascending=True).reset_index(drop=True)
        num_train = int(len(df_num) * 0.7)
        num_test = int(len(df_num) * 0.2)
        num_vali = len(df_num) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_num) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_num)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        
        first_start_date = df_num.start_date[border1]
        final_end_date = df_num.end_date[border2-1]

        df_data = df_num[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values).astype(np.float32)
            self.mean_data = self.scaler.mean_
            self.std_data = self.scaler.scale_
        else:
            data = df_data.values.astype(np.float32)

        df_stamp = df_num[['date']][border1:border2]
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]


        self.data_stamp = data_stamp
        self.num_dates = df_num[['start_date', 'end_date']][border1:border2].reset_index(drop=True)
        self.txt_report = df_report[['start_date', 'end_date', 'fact']].loc[(df_report.end_date >= first_start_date) & (df_report.end_date <= final_end_date)]
        self.txt_search = df_search[['start_date', 'end_date', 'fact']] if not df_search.empty else df_search

    def collect_text(self, start_date, end_date, seq_x=None, cache_key=None):
        if self.rag_cache_guidance and cache_key is not None and cache_key in self.guidance_cache:
            return self.guidance_cache[cache_key]

        report = self.txt_report.loc[(self.txt_report.end_date >= start_date) & (self.txt_report.end_date <= end_date)]
        def add_datemark(row):
            return row['start_date'].strftime("%Y-%m-%d") + " to " + row['end_date'].strftime("%Y-%m-%d") + ": " + row['fact']
        if not report.empty:
            report = report.apply(add_datemark, axis=1).to_list()
            report.insert(0, self.desc)
            text_mark = 1
        else:
            report = ['NA']
            text_mark = 0
        all_txt = ' '.join(report)
        if self.rag_cot is not None:
            raw_txt = all_txt if text_mark else ''
            search_pool = self.txt_search.loc[self.txt_search.end_date <= end_date] if not self.txt_search.empty else self.txt_search
            all_txt = self.rag_cot.build_guidance_text(raw_txt, seq_x, self.desc, search_frame=search_pool)
            text_mark = 1 if all_txt and all_txt != 'NA' else text_mark
        if self.rag_cache_guidance and cache_key is not None:
            self.guidance_cache[cache_key] = (all_txt, text_mark)
        return all_txt, text_mark
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end, :]
        seq_y = self.data_y[r_begin:r_end, :]
    
        seq_x_stamp = self.data_stamp[s_begin:s_end]
        seq_y_stamp = self.data_stamp[r_begin:r_end]

        text_begin = s_end - self.text_len
        text_end = s_end

        seq_x_txt, txt_mark = self.collect_text(self.num_dates.start_date[text_begin], self.num_dates.end_date[text_end], seq_x, cache_key=index)

        observed_data = np.concatenate([seq_x, seq_y], axis=0)
        timesteps = np.concatenate([seq_x_stamp, seq_y_stamp], axis=0)
        observed_mask = np.ones_like(observed_data)
        gt_mask = np.concatenate([np.ones_like(seq_x), np.zeros_like(seq_y)], axis=0)

        s = {
            'observed_data': observed_data,
            'observed_mask': observed_mask,
            'gt_mask': gt_mask,
            'timepoints': np.arange(self.seq_len + self.pred_len).astype(np.float32), 
            'feature_id': np.arange(seq_x.shape[1]).astype(np.float32),
            'timesteps': timesteps,
            'texts': seq_x_txt,
            'text_mark': txt_mark
        }

        return s

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv', target='OT',
                 scale=True, timeenc=0, freq='h', text_len=None,
                 scaler_type='minmax'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[-1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        if scale:
            if scaler_type == 'minmax':
                self.scaler = MinMaxScaler(feature_range=(-1, 1))
            elif scaler_type == 'standard':
                self.scaler = StandardScaler()
            else:
                scaler_type = 'minmax'
                self.scaler = MinMaxScaler(feature_range=(-1, 1))
        else:
            self.scaler = None
        self.scaler_type = scaler_type

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values).astype(np.float32)
        else:
            data = df_data.values.astype(np.float32)

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv', target='OT',
                 scale=True, timeenc=0, freq='t', text_len=None,
                 scaler_type='minmax'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[-1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        if scale:
            if scaler_type == 'minmax':
                self.scaler = MinMaxScaler(feature_range=(-1, 1))
            elif scaler_type == 'standard':
                self.scaler = StandardScaler()
            else:
                scaler_type = 'minmax'
                self.scaler = MinMaxScaler(feature_range=(-1, 1))
        else:
            self.scaler = None
        self.scaler_type = scaler_type
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values).astype(np.float32)
        else:
            data = df_data.values.astype(np.float32)

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        observed_data = np.concatenate(seq_x, seq_y, axis=0)
        timesteps = np.concatenate(seq_x_mark, seq_y_mark, axis=0)
        observed_mask = np.ones_like(observed_data)
        gt_mask = np.concatenate(np.ones_like(seq_x), np.zeros_like(seq_y), axis=0)

        s = {
            'observed_data': observed_data,
            'observed_mask': observed_mask,
            'gt_mask': gt_mask,
            'timepoints': np.arange(self.seq_len + self.pred_len) * 1.0, 
            'feature_id': np.arange(self.seq_x.shape[1]) * 1.0,
            'timesteps': timesteps

        }

        return s

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
