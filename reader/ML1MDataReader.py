from reader.BaseReader import BaseDataReader
import numpy as np
from utils import padding_and_clip
class ML1MDataReader(BaseDataReader):

    def __init__(self, params):
        '''
        - from BaseReader:
            - phase
            - data: will add Position column
        '''
        super().__init__(params)
        self.max_seq_len = params['max_seq_len']

    def _read_data(self, params):
        # read data_file
        super()._read_data(params)
        print("Load item meta data")
        self.item_meta = params['item_meta']
        self.user_meta = params['user_meta']
        self.item_vec_size = len(self.item_meta[0])
        self.user_vec_size = len(self.user_meta[0])
        self.portrait_len = len(self.user_meta[0])

    ###########################
    #        Iterator         #
    ###########################

    def __getitem__(self, idx):
        user_ID, slate_of_items, user_feedback, user_history, sequence_id = self.data[self.phase].iloc[idx]
        user_profile = self.user_meta[user_ID]

        exposure = eval(slate_of_items)

        history = eval(user_history)

        hist_length = len(history)
        history = padding_and_clip(history, self.max_seq_len)
        # print(f"history{}")
        feedback = eval(user_feedback)

        record = {
            'timestamp': int(1),  # timestamp is irrelevant, just a hack temporal
            'exposure': np.array(exposure).astype(int),
            'exposure_features': self.get_item_list_meta(exposure).astype(float),
            'feedback': np.array(feedback).astype(float),
            'history': np.array(history).astype(int),
            'history_features': self.get_item_list_meta(history).astype(float),
            'history_length': int(min(hist_length, self.max_seq_len)),
            'user_profile': np.array(user_profile)
        }
        return record

    def get_item_list_meta(self, item_list):
        return np.array([self.item_meta[item] for item in item_list])

    def get_statistics(self):
        '''
        - n_user
        - n_item
        - s_parsity
        - from BaseReader:
            - length
            - fields
        '''
        stats = super().get_statistics()
        stats['length'] = len(self.data[self.phase])
        stats['n_item'] = len(self.item_meta) - 1
        stats['item_vec_size'] = self.item_vec_size
        stats['user_portrait_len'] = self.user_vec_size
        stats['max_seq_len'] = self.max_seq_len
        return stats