class CategoricalVariableCombiner():
    def __init__(self, cover_freq=0.95, max_components=100, counter=None):
        self.cover_freq = cover_freq
        self.max_components = max_components
        self.counter = counter
        self.use_hash = False
        self.mapper = None
        self.map_func = None

    def fit(self, X):
        if X.ndim != 1:
            raise ValueError('Unexpected input dimension %d, expected %d' % (X.ndim, 1,))
        n_samples = len(X)
        nb_categories = len(set(X))
        self.data_type = X.dtype
        if self.data_type == object:
            self.unknown = 'UNKNOWN'
        else:
            self.unknown = 99999999
        if self.counter is None:
            self.counter = Counter(X)
            self.counter = dict((key, self.counter[key] / float(n_samples)) for key in self.counter.keys())
        self.counter = OrderedDict(
            (key, value) for key, value in sorted(self.counter.iteritems(), key=lambda x : (x[1], x[0]), reverse=True))
        accum, n_component, keys = 0.0, 0, self.counter.keys()
        while n_component < self.max_components:
            accum += self.counter[keys[n_component]]
            if accum >= self.cover_freq:
                break
            else:
                n_component += 1
        if n_component < self.max_components:
            self.mapper = {}
            for i, key in enumerate(keys):
                # cut_off_components = nb_categories if nb_categories < self.max_components else n_component
                cut_off_components = n_component
                if i <= cut_off_components:
                    self.mapper[key] = key
                else:
                    self.mapper[key] = self.unknown
        else:
            self.use_hash = True
            self.hash_trick(X, self.max_components, hash_seed=42)

        if self.mapper is not None:
            self.map_func = np.vectorize(lambda x: self.mapper[x], otypes=[self.data_type])

        return self

    def transform(self, X):
        if X.dtype != self.data_type:
            raise ValueError('Unexpected input data type %s, expected %s' % (X.dtype, self.data_type,))
        if self.mapper is None:
            raise ValueError('Must train combiner before it can be used to transform data.')
        X_encoded = X.copy()
        if self.map_func is not None:
            mask = np.vectorize(lambda x: x in self.mapper)(X_encoded)
            X_encoded[mask] = self.map_func(X_encoded[mask])
            if self.use_hash:
                X_encoded[~mask] = self.mapper[self.unknown]
            else:
                X_encoded[~mask] = self.unknown

        return X_encoded

    def hash_trick(self, X, max_components, hash_seed=42):
        categories = set(X)
        self.mapper = {}
        for cat in categories:
            self.mapper[cat] = int(hashlib.sha1(str(cat)).hexdigest(), 16) % max_components
        self.mapper[self.unknown] = int(hashlib.sha1(str(self.unknown)).hexdigest(), 16) % max_components


class CategoricalEncoder():
    def __init__(self):
        self.classes = 0
        self.class_ = {}

    def fit(self, y):
        unique_vals = set(y)
        self.classes = len(unique_vals)
        self.class_ = {k: v for v, k in enumerate(unique_vals)}

        return self

    def fit_transform(self, y):
        unique_vals = set(y)
        self.classes = len(unique_vals)
        self.class_ = {k: v for v, k in enumerate(unique_vals)}
        return np.vectorize(lambda x: self.class_[x])(y)

    def transform(self, y):
        def mapper(x):
            if x in self.class_:
                return self.class_[x]
            else:
                self.class_[x] = self.classes
                self.classes += 1
                return self.class_[x]

        return np.vectorize(mapper)(y)

    def inverse_transform(self, y):
        classes = np.unique(y)
        if len(np.intersect1d(classes, self.class_.values())) < len(classes):
            diff = np.setdiff1d(classes, self.class_.values())
            raise ValueError("y contains new labels: %s" % str(diff))
        self.inverse_class_ = {v: k for k, v in self.class_.iteritems()}
        return np.vectorize(lambda x: self.inverse_class_[x])(y)


class RecordField():
    def __init__(self, timedelta, label):
        self.timedelta = timedelta
        self.label = label
        self.sum = 0.0
        self.nb_records = 0
        self.squared_sum = 0.0
        self.min = np.inf
        self.max = -np.inf

    def put(self, label):
        self.label += label

    def update(self, value):
        self.sum += value
        self.squared_sum += (value ** 2)
        self.nb_records += 1
        self.min = np.min([self.min, value])
        self.max = np.max([self.max, value])

    def get_stats(self, stat_type='mean'):
        if self.nb_records == 0:
            return np.nan
        else:
            if stat_type == 'mean':
                #eturn float(self.sum)
                return float(self.sum) / self.nb_records
            elif stat_type == 'std':
                return float(self.squared_sum - float(self.sum ** 2) / self.nb_records) / self.nb_records
            elif stat_type == 'min':
                return float(self.min)
            elif stat_type == 'max':
                return float(self.max)
            else:
                raise ValueError('Unsupported Statistic Type!')

    def get_label(self):
        return self.label

    def get_nb_records(self):
        return self.nb_records

    def get_period(self):
        return self.timedelta


class HistoricalFeatureAggregator():
    def __init__(self, datetime_col, categorical_cols, datetime_categorical_cols, label, candidate_periods,
                 stats_func=['mean'], max_len=2, skip_window=np.timedelta64(0, 'D')):
        self.datetime_col = datetime_col
        self.categorical_cols = categorical_cols
        self.datetime_categorical_cols = datetime_categorical_cols
        self.label = label
        self.candidate_periods = candidate_periods
        self.stats_func = stats_func
        self.max_len = max_len
        self.skip_window = skip_window
        self.category_combinations = None
        self.historical_moving_data = None
        self.col_index = None

    def partial_fit(self, X):
        self.historical_moving_data = OrderedDict()
        self.category_combinations = self.generate_all_combinations(self.categorical_cols, max_len=self.max_len)
        cross_category_combinations = self.cross_product_two_lists(self.category_combinations,
                                                                   self.datetime_categorical_cols)
        self.category_combinations.extend(cross_category_combinations)
        self.col_index = {key: value + 1 for value, key in enumerate(X.columns.values)}
        start_time = time.time()
        idx = 0
        for row in X.itertuples():
            if (idx % 100000 == 0) & (idx != 0):
                print 'Processed {}% data'.format(100.0 * float(idx) / X.shape[0])
                end_time = time.time()
                print 'Take %s seconds to process 100000 records' % (end_time - start_time)
                start_time = end_time
            for combo in self.category_combinations:
                for period in self.get_periods(combo):
                    cols = [row[self.col_index[i]] for i in combo]
                    feat_name = '_'.join(combo + ['_'.join(str(period).split())])
                    key_name = feat_name + '_' + '_'.join([str(x) for x in cols])
                    if row[self.col_index[self.datetime_col]] not in self.historical_moving_data:
                        self.historical_moving_data[row[self.col_index[self.datetime_col]]] = {}
                    if key_name not in self.historical_moving_data[row[self.col_index[self.datetime_col]]]:
                        self.historical_moving_data[row[self.col_index[self.datetime_col]]][key_name] = RecordField(period, row[self.col_index[self.label]])
                    else:
                        self.historical_moving_data[row[self.col_index[self.datetime_col]]][key_name].put(
                            row[self.col_index[self.label]])

            idx += 1

    def fit(self, X):
        """
        X: pandas dataframe object
        """
        self.partial_fit(X)
        keys = self.historical_moving_data.keys()
        key_len = len(keys)
        start_time = time.time()
        """
        for i in range(key_len - 1):
            j = i + 1
            if (i % 100 == 0) & (i != 0):
                #end_time = time.time()
                print 'Processed {} % data'.format(100.0 * float(i) / key_len)
                #print 'Take %s seconds for 10 records' % (end_time - start_time)
                #start_time = end_time
            
            for k in self.historical_moving_data[keys[i]]:
                period = self.historical_moving_data[keys[i]][k].get_period()
                j = i + 1
                while (j < (key_len - 1)) and (keys[i] - keys[j] <= period + self.skip_window):
                    if keys[i] - keys[j] > self.skip_window and k in self.historical_moving_data[keys[j]]:
                        self.historical_moving_data[keys[i]][k].update(
                            self.historical_moving_data[keys[j]][k].get_label())

                    j += 1
            
            max_period = np.max([x.get_period() for x in self.historical_moving_data[keys[i]].values()])
            candidate_keys = self.historical_moving_data[keys[i]].keys()
            while (j < (key_len - 1)) and (keys[i] - keys[j] <= max_period + self.skip_window):
                time_delta = keys[i] - keys[j]
                if time_delta > self.skip_window:
                    inter_keys = np.intersect1d(candidate_keys, self.historical_moving_data[keys[j]].keys())
                    for k in inter_keys:
                        if (time_delta<=self.historical_moving_data[keys[i]][k].get_period()+self.skip_window):
                            self.historical_moving_data[keys[i]][k].update(self.historical_moving_data[keys[j]][k].get_label())
                j += 1
            

        """
        inputs = [(self, (i, keys, key_len)) for i in range(key_len - 1)]
        p = Pool()
        results = p.map(process, inputs)
        for k, v in results:
            self.historical_moving_data[k] = v
        end_time = time.time()
        print 'Take %s seconds to complete' % (end_time - start_time)

        return self

    def transform(self, X):
        """
        X: pandas dataframe object
        """
        if self.category_combinations is None:
            raise ValueError('Must train combiner before it can be used to transform data.')
        historical_agg_features = {}
        start_time = time.time()
        idx = 0
        for row in X.itertuples():
            if (idx % 100000 == 0) & (idx != 0):
                print 'Processed {} % data'.format(100.0 * float(idx) / X.shape[0])
                end_time = time.time()
                print 'Take %s seconds to process 100000 records' % (end_time - start_time)
                start_time = end_time
            for combo in self.category_combinations:
                for period in self.get_periods(combo):
                    cols = [row[self.col_index[i]] for i in combo]
                    feat_name = '_'.join(combo + ['_'.join(str(period).split())])
                    key_name = feat_name + '_' + '_'.join([str(x) for x in cols])
                    if row[self.col_index[self.datetime_col]] in self.historical_moving_data:
                        for s in self.stats_func:
                            col_name = feat_name + '_' + s
                            if col_name not in historical_agg_features:
                                historical_agg_features[col_name] = []
                            historical_agg_features[col_name].append(
                                self.historical_moving_data[row[self.col_index[self.datetime_col]]][key_name].get_stats(
                                    s))
                    else:
                        keys = np.array(self.historical_moving_data.keys())
                        available_keys = keys[(keys < (row[self.col_index[self.datetime_col]] - self.skip_window)) & (
                            keys >= (row[self.col_index[self.datetime_col]] - self.skip_window - period))]
                        find_key = False
                        for k in available_keys:
                            if key_name in self.historical_moving_data[k]:
                                find_key = True
                                break
                        for s in self.stats_func:
                            col_name = feat_name + '_' + s
                            if col_name not in historical_agg_features:
                                historical_agg_features[col_name] = []
                            if find_key:
                                historical_agg_features[col_name].append(
                                    self.historical_moving_data[k][key_name].get_stats(s))
                            else:
                                historical_agg_features[col_name].append(np.nan)

            idx += 1
        historical_features_df = pd.DataFrame(historical_agg_features)

        return historical_features_df

    @staticmethod
    def generate_all_combinations(categorical_variables, max_len):
        res = []
        for i in range(1, min(max_len, len(categorical_variables)) + 1):
            for subset in combinations(categorical_variables, i):
                res.append(list(subset))
        return res

    @staticmethod
    def cross_product_two_lists(list_A, list_B):
        """
        list_A : list of list, [['a'], ['b'], ['c']]
        list_B : list, ['d', 'e', 'f']

        return : list [['a', 'd'], ['a', 'e'], ['a', 'f'], ['b', 'd'], ['b', 'e'], ['b', 'f'],
                   ['c', 'd'], ['c', 'e'], ['c', 'f']]
        """
        res = []
        for prod in product(list_A, list_B):
            if prod[0]:
                tmp = prod[0] + [prod[1]]
            else:
                tmp = [prod[1]]
            res.append(tmp)
        return res


    def get_periods(self, categorical_variables):
        intersection = np.intersect1d(categorical_variables, self.datetime_categorical_cols)
        if len(intersection) > 1:
            raise ValueError('Do not support two datetime categorical variables!')
        elif len(intersection) == 1:
            return self.candidate_periods[intersection[0]]
        else:
            if 'other' in self.candidate_periods:
                return self.candidate_periods['other']
            else:
                raise ValueError('Please sepecify the periods for other categorical variables!')

def process(x):
    self = x[0]
    i, keys, key_len = x[1][0], x[1][1], x[1][2]
    j = i + 1
    print 'Processed {} % data'.format(100.0 * float(i) / key_len)
    max_period = np.max([x.get_period() for x in self.historical_moving_data[keys[i]].values()])
    candidate_keys = self.historical_moving_data[keys[i]].keys()
    while (j < (key_len - 1)) and (keys[i] - keys[j] <= max_period + self.skip_window):
        time_delta = keys[i] - keys[j]
        if time_delta > self.skip_window:
            inter_keys = np.intersect1d(candidate_keys, self.historical_moving_data[keys[j]].keys())
            for k in inter_keys:
                if (time_delta<=self.historical_moving_data[keys[i]][k].get_period()+self.skip_window):
                    self.historical_moving_data[keys[i]][k].update(self.historical_moving_data[keys[j]][k].get_label())
        j += 1
    return (keys[i], self.historical_moving_data[keys[i]])