import h5py


class SkipthBatchGenerator:
    def __init__(self, filename):
        """
        :param filename:
        :param np_data: should be a return value from
        :param maxlength:
        :return:
        """
        self.filename = filename
        self.file = None
        self.np_data = None
        self.epoch = 0
        self.cur_num = 0
        self.total_num = 0
        self.total_batch = 0
        self.cur_pointer = 0

    def load_data(self):
        self.file = h5py.File(self.filename, "r")
        self.np_data = self.file['premise'], self.file['hypothesis'], self.file['label']
        self.total_num = self.file['premise'].shape[0]
        if self.file['premise'].shape[0] == self.file['hypothesis'].shape[0] == self.file['label'].shape[0]:
            print(self.filename, 'verified')
        return self.total_num # return total number of data

    def next_batch(self, batch_size, **kwargs):
        if self.np_data is None:
            print('Data not loaded.')
            return None

        if batch_size == -1:
            premise, hypothesis, label = self.np_data
            return premise[:], hypothesis[:], label[:]

        start = self.cur_pointer
        end = self.cur_pointer + batch_size
        if not end < self.total_num:
            end = self.total_num
            self.epoch += 1
            self.cur_pointer = 0
        else:
            self.cur_pointer = end
        self.total_batch += 1
        premise, hypothesis, label = self.np_data
        return premise[start:end], hypothesis[start:end], label[start:end]

    def get_epoch(self, batch_size):
        if self.np_data is None:
            print('Data not loaded')
            return
        cur_epoch = self.epoch
        while cur_epoch == self.epoch:
            yield self.next_batch(batch_size)


class BatchGeneratorH5:
    def __init__(self, filename, maxlength):
        """
        :param filename:
        :param np_data: should be a return value from
        :param maxlength:
        :return:
        """
        self.filename = filename
        self.file = None
        self.np_data = None
        self.epoch = 0
        self.cur_num = 0
        self.total_num = 0
        self.total_batch = 0
        self.cur_pointer = 0
        self.maxlength = maxlength

    def load_data(self):
        self.file = h5py.File(self.filename, "r")
        self.np_data = self.file['premise'], self.file['p_len'], \
                       self.file['hypothesis'], self.file['h_len'], self.file['label']
        self.total_num = self.file['premise'].shape[0]
        if self.file['premise'].shape[0] == self.file['hypothesis'].shape[0] == self.file['label'].shape[0] \
                == self.file['p_len'].shape[0] == self.file['h_len'].shape[0]:
            print(self.filename, 'verified')
        return self.total_num  # return total number of data

    def next_batch(self, batch_size, **kwargs):
        if self.np_data is None:
            print('Data not loaded.')
            return None

        if batch_size == -1:
            premise, p_len, hypothesis, h_len, label = self.np_data
            return premise[:, :self.maxlength], p_len[:], hypothesis[:, :self.maxlength], h_len[:], label[:]

        start = self.cur_pointer
        end = self.cur_pointer + batch_size
        if not end < self.total_num:
            end = self.total_num
            self.epoch += 1
            self.cur_pointer = 0
        else:
            self.cur_pointer = end
        self.total_batch += 1
        premise, p_len, hypothesis, h_len, label = self.np_data
        return premise[start:end, :self.maxlength], p_len[start:end], hypothesis[start:end, :self.maxlength], h_len[start:end], label[start:end]

    def get_epoch(self, batch_size):
        if self.np_data is None:
            print('Data not loaded')
            return
        cur_epoch = self.epoch
        while cur_epoch == self.epoch:
            yield self.next_batch(batch_size)


if __name__ == '__main__':
    pass
    # import config
    # test_gen = SkipthBatchGenerator(config.SNLI_ST_TEST_FILE)
    # test_gen.load_data()
    # a, b, c = test_gen.next_batch(-1)
    # print(a.shape, b, c)
    #
    # test_gen_b = BatchGeneratorH5(config.SNLI_TEST_FILE, maxlength=80)
    # test_gen_b.load_data()
    # a, b, c, d, e, = test_gen_b.next_batch(-1)
    # print(a.shape)
    #
    # for i in test_gen_b.get_epoch(batch_size=2000):
    #     print(len(i[0]))
