from tensorflow.keras.layers import Input, LSTM, Attention, Dense, Concatenate, TimeDistributed, LeakyReLU, Bidirectional
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.backend import clear_session
import numpy as np


'''
E1: 해당 주의 확진자 데이터만 사용
'''
class E1():
    def __init__(self, **kwargs):
        clear_session()
        
        ENC_LSTM_UNIT = 128
        DEC_LSTM_UNIT = 128
        DROPOUT = 0.2
        
        self.look_back = kwargs['look_back']
        self.look_ahead = kwargs['look_ahead']
        
        ### 학습 과정 ###
        enc1_inputs = Input(shape=(self.look_back, 1), name='enc1_input')
        enc1_lstm1 = LSTM(ENC_LSTM_UNIT, return_state=True, return_sequences=True,
                          dropout=DROPOUT, name='enc1_lstm1')
        enc1_lstm2 = LSTM(ENC_LSTM_UNIT, return_state=True, return_sequences=True,
                          dropout=DROPOUT, name='enc1_lstm2')
        enc1_outputs, _, _ = enc1_lstm1(enc1_inputs)
        enc1_outputs, enc1_h, enc1_c = enc1_lstm2(enc1_outputs)
        
        dec_inputs = Input(shape=(None, 1), name='dec_input')
        dec_lstm1 = LSTM(DEC_LSTM_UNIT, return_state=True, return_sequences=True,
                         dropout=DROPOUT, name='dec_lstm1')
        dec_outputs, _, _ = dec_lstm1(dec_inputs, initial_state=[enc1_h, enc1_c])
        
        dec_dense1 = TimeDistributed(Dense(1, name='dense1'), name='time1')
        outputs = dec_dense1(dec_outputs)
        
#         dec_attn = Attention(name='dec_attn')([dec_outputs, enc1_outputs])
#         dec_concat = Concatenate(name='dec_concat')([dec_outputs, dec_attn])
        
#         dec_dense1 = TimeDistributed(Dense(1, name='dense1'), name='time1')
#         outputs = dec_dense1(dec_concat)
        
        model = Model([enc1_inputs, dec_inputs], outputs)
        model.compile(optimizer='adam',
                      loss='mse')
        
        self.model = model
        
        ### 예측 과정 ###
        # 예측 과정에서 decoder는 states를 입력 받아야함
        dec_state_inputs = [
            Input(shape=(DEC_LSTM_UNIT,), name='dec_h_input'),
            Input(shape=(DEC_LSTM_UNIT,), name='dec_c_input')
        ]
        
        dec_hidden_inputs = Input(shape=(self.look_back, DEC_LSTM_UNIT), name='dec_hidden_input')
        
        # 기존 decoder와 동일함
        dec_outputs2, dec_state_h, dec_state_c = dec_lstm1(dec_inputs,
                                                           initial_state=dec_state_inputs)
        dec_outputs2 = dec_dense1(dec_outputs2)
#         dec_attn2 = Attention(name='dec_test_attn2')([dec_outputs2, dec_hidden_inputs])
#         dec_concat2 = Concatenate(name='dec_test_concat2')([dec_outputs2, dec_attn2])
#         dec_outputs2 = dec_dense1(dec_concat2)
        
        # 예측 과정에서 encoder, decoder를 분리해서 사용함.
        # 학습 과정과 예측 과정에서 이렇게 차이가 발생하는 이유는
        # 학습 과정에서는 교사 강요를 사용해 예측 값을 다시 입력으로 넣지 않지만,
        # 예측 과정에서는 예측 값을 입력으로 다시 넣기 때문.
        self.encoder_model = Model(enc1_inputs, [enc1_h, enc1_c])
        self.decoder_model = Model([dec_inputs] + dec_state_inputs,
                                   [dec_outputs2] + [dec_state_h, dec_state_c])
        
#         self.encoder_model = Model(enc1_inputs, [enc1_outputs, enc1_h, enc1_c])
#         self.decoder_model = Model([dec_inputs] + [dec_hidden_inputs] + dec_state_inputs,
#                                    [dec_outputs2] + [dec_state_h, dec_state_c])
        
    def fit(self, x, y, epochs=10, batch_size=32, validation_split=0.2, callbacks=None,
            shuffle=False, verbose=1):
        history = self.model.fit(x, y, epochs=epochs, batch_size=batch_size,
                                validation_split=validation_split, callbacks=callbacks,
                                shuffle=shuffle, verbose=verbose)
        return history
    
    def load_model(self, model_path=None):
        self.model = load_model(model_path)
        
        # model을 load한 후에, 
        # encoder_model, decoder_model 에 weight 설정
        self.encoder_model.get_layer('enc1_lstm1').set_weights(
            self.model.get_layer('enc1_lstm1').get_weights())
        self.encoder_model.get_layer('enc1_lstm2').set_weights(
            self.model.get_layer('enc1_lstm2').get_weights())
        self.decoder_model.get_layer('dec_lstm1').set_weights(
            self.model.get_layer('dec_lstm1').get_weights())
        self.decoder_model.get_layer('time1').set_weights(
            self.model.get_layer('time1').get_weights())  
        
    def generate_data(self, truth, test_split_size=0.8):
        encoder_inputs = []
        decoder_inputs = []
        y_hat = []
        
        # encoder의 경우 [y_t-lb, y_t-lb+1, ..., y_t-1] 사용
        # decoder의 경우 [y_t, y_t+1, ..., y_t+la-1] 사용
        # y_hat(학습의 답)의 경우 [y_t+1, y_t+2, ..., y_t+la] 사용
        # 여기서 lb 는 look back, 예측을 위해 사용될 "과거" 데이터 길이
        # la는 look ahead, 몇 단계를 연속 예측할지 정하는 값
        for i in range(len(truth) - self.look_back - self.look_ahead):
            encoder_inputs.append(truth[i:i+self.look_back])
            decoder_inputs.append(truth[i+self.look_back-1:i+self.look_back+self.look_ahead-1])
            y_hat.append(truth[i+self.look_back:i+self.look_back+self.look_ahead])
            
        encoder_inputs = np.array(encoder_inputs) + np.random.uniform(1e-6, 1e-5, size=(len(encoder_inputs), self.look_back, 1))
        decoder_inputs = np.array(decoder_inputs)
        y_hat = np.array(y_hat)
        
        test_size = int(len(truth) * test_split_size)
        
        x_train = [encoder_inputs[:test_size], decoder_inputs[:test_size]]
        x_test = [encoder_inputs[test_size:], decoder_inputs[test_size:]]
        x_total = [encoder_inputs, decoder_inputs]
        
        y_train = y_hat[:test_size]
        y_test = y_hat[test_size:]
        
        return [x_train, x_test, x_total], [y_train, y_test, y_hat], test_size 
    
    def predict(self, enc_inputs, dec_inputs):
        assert len(enc_inputs) == len(dec_inputs), 'not equal length of enc_inputs(%s) and enc_inputs(%s)' % (len(enc_inputs), len(dec_inputs))
        
        prd_values = []
        # input의 길이만큼 반복
        for i in range(len(enc_inputs)):
            # encoder만 사용해서 encoder의 output과 states 얻기
            enc_h, enc_c = self.encoder_model.predict(enc_inputs[i].reshape(1, self.look_back, 1))
#             enc_out, enc_h, enc_c = self.encoder_model.predict(enc_inputs[i].reshape(1, self.look_back, 1))

            # decoder input의 0번 째 값만 사용하기
            # 왜냐하면 위에서 decoder input을 만들 때 
            # 학습용으로 만들었기 때문에
            # [[y_t, y_t+1, ...], [y_t+1, y_t+2, ...]] 이런식의 형태를 가짐
            # 따라서 각 리스트 첫 번째 값(실제 값)만 처음으로 입력하기 위함
            dec_in = dec_inputs[i][0].reshape(-1, 1)
            
            tmp = []
            # 이후는 look ahead만큼 연속적으로 예측한다.
            for ahead in range(self.look_ahead):
                # decoder는 받은 첫 번째 값으로 예측 및 states를 반환
                prd, dec_h, dec_c = self.decoder_model.predict([dec_in] + [enc_h, enc_c])
#                 prd, dec_h, dec_c = self.decoder_model.predict([dec_in] + [enc_out, enc_h, enc_c])
                prd_val = prd[0, 0, 0]
                
                if prd_val > 0:
                    tmp.append(prd_val)
                else:
                    # 예측 값이 0보다 작으면 아주 작은 값을 줌
                    # 0일 경우 MAPE, CORR 계산에서 오류가 날 수 있음
                    tmp.append(np.random.uniform(1e-6, 1e-5))
                
                # 연속적인 예측을 위해 다음 decoder input을 예측된 값으로 설정
                dec_in = prd[0, 0]
                # decoder에 들어갈 states를 반환된 states로 설정
                enc_h, enc_c = dec_h, dec_c
            
            prd_values.append(tmp)
        prd_values = np.array(prd_values)
        return prd_values  
    
    
'''
E2: 해당 주 확진자 + 인접 주 확진자 데이터 사용
'''
class E2():
    def __init__(self, **kwargs):
        DENSE_UNIT = 64
        ENC_LSTM_UNIT = 128
        DEC_LSTM_UNIT = ENC_LSTM_UNIT * 2 * 2
        DROPOUT = 0.2
        
        self.look_back = kwargs['look_back']
        self.look_ahead = kwargs['look_ahead']
        self.num_adjacent = kwargs['num_adjacent']
        
        ### 학습 과정 ###
        # encoder 1
        enc1_inputs = Input(shape=(self.look_back, 1), name='enc1_input')
        
        # LSTM 2개 사용
        enc1_lstm1 = Bidirectional(LSTM(ENC_LSTM_UNIT, return_state=True, return_sequences=True,
                          dropout=DROPOUT, name='enc1_lstm1'), name='enc1_bilstm1')
        enc1_lstm2 = Bidirectional(LSTM(ENC_LSTM_UNIT, return_state=True, return_sequences=True,
                          dropout=DROPOUT, name='enc1_lstm2'), name='enc1_bilstm2')
        enc1_outputs, _, _, _, _ = enc1_lstm1(enc1_inputs)
        enc1_outputs, enc1_fh, enc1_fc, enc1_bh, enc1_bc = enc1_lstm2(enc1_outputs)
        enc1_h = Concatenate(name='enc1_h')([enc1_fh, enc1_bh])
        enc1_c = Concatenate(name='enc1_c')([enc1_fc, enc1_bc])
        
        # encoder 2
        enc2_inputs = Input(shape=(self.look_back, self.num_adjacent), name='enc2_input')
        enc2_lstm1 = Bidirectional(LSTM(ENC_LSTM_UNIT, return_state=True, return_sequences=True,
                          dropout=DROPOUT, name='enc2_lstm1'), name='enc2_bilstm1')
        enc2_lstm2 = Bidirectional(LSTM(ENC_LSTM_UNIT, return_state=True, return_sequences=True,
                          dropout=DROPOUT, name='enc2_lstm2'), name='enc2_bilstm2')
        enc2_outputs, _, _, _, _ = enc2_lstm1(enc2_inputs)
        enc2_outputs, enc2_fh, enc2_fc, enc2_bh, enc2_bc = enc2_lstm2(enc2_outputs)
        enc2_h = Concatenate(name='enc2_h')([enc2_fh, enc2_bh])
        enc2_c = Concatenate(name='enc2_c')([enc2_fc, enc2_bc])
        
        
        enc_outputs = Concatenate(axis=-1, name='enc_outputs_concat')([enc1_outputs, enc2_outputs])
        enc_states = [
            Concatenate(name='enc_h')([enc1_h, enc2_h]),
            Concatenate(name='enc_c')([enc1_c, enc2_c]),
        ]
        
        
        # decoder의 input shape. (None, 1)
        dec_inputs = Input(shape=(None, 1), name='dec_input')
        dec_lstm1 = LSTM(DEC_LSTM_UNIT, return_state=True, return_sequences=True,
                         dropout=DROPOUT, name='dec_lstm1')
        dec_outputs, _, _ = dec_lstm1(dec_inputs, initial_state=enc_states)
        
        dec_attn = Attention(name='dec_attn')([dec_outputs, enc_outputs])
        dec_concat = Concatenate(name='dec_concat')([dec_outputs, dec_attn])
        
        dec_dense1 = TimeDistributed(Dense(DENSE_UNIT, name='dense1'), name='time1')
        dec_dense2 = TimeDistributed(Dense(1, name='dense2'), name='time2')
        leaky_relu = LeakyReLU(alpha=0.1)
        outputs = dec_dense1(dec_concat)
        outputs = dec_dense2(outputs)
        outputs = leaky_relu(outputs)
        
        model = Model([enc1_inputs, enc2_inputs, dec_inputs], outputs)
        model.compile(optimizer='adam',
                      loss='mse')
        
        self.model = model
        
        ### 예측 과정 ###
        # 예측 과정에서 decoder는 states를 입력 받아야함
        dec_state_inputs = [
            Input(shape=(DEC_LSTM_UNIT,), name='dec_h_input'),
            Input(shape=(DEC_LSTM_UNIT,), name='dec_c_input')
        ]
        dec_hidden_inputs = Input(shape=(self.look_back, DEC_LSTM_UNIT), name='dec_hidden_input')
        
        dec_outputs2, dec_state_h, dec_state_c = dec_lstm1(dec_inputs, 
                                                           initial_state=dec_state_inputs)
        dec_attn2 = Attention(name='dec_test_attn2')([dec_outputs2, dec_hidden_inputs])
        dec_concat2 = Concatenate(name='dec_test_concat2')([dec_outputs2, dec_attn2])
        dec_outputs2 = dec_dense1(dec_concat2)
        dec_outputs2 = dec_dense2(dec_outputs2)
        dec_outputs2 = leaky_relu(dec_outputs2)
        
        self.encoder_model = Model([enc1_inputs, enc2_inputs], [enc_outputs] + enc_states)
        self.decoder_model = Model([dec_inputs] + [dec_hidden_inputs] + dec_state_inputs,
                                   [dec_outputs2] + [dec_state_h, dec_state_c])
        
    def fit(self, x, y, epochs=10, batch_size=32, validation_split=0.2, callbacks=None,
            shuffle=False, verbose=1):
        history = self.model.fit(x, y, epochs=epochs, batch_size=batch_size,
                                validation_split=validation_split, callbacks=callbacks,
                                shuffle=shuffle, verbose=verbose)
        return history
    
    def load_model(self, model_path=None):
        self.model = load_model(model_path)
        
        # model을 load한 후에, 
        # encoder_model, decoder_model 에 weight 설정
        self.encoder_model.get_layer('enc1_bilstm1').set_weights(
            self.model.get_layer('enc1_bilstm1').get_weights())
        self.encoder_model.get_layer('enc1_bilstm2').set_weights(
            self.model.get_layer('enc1_bilstm2').get_weights())
        self.encoder_model.get_layer('enc2_bilstm1').set_weights(
            self.model.get_layer('enc2_bilstm1').get_weights())
        self.encoder_model.get_layer('enc2_bilstm2').set_weights(
            self.model.get_layer('enc2_bilstm2').get_weights())
        self.decoder_model.get_layer('dec_lstm1').set_weights(
            self.model.get_layer('dec_lstm1').get_weights())
        self.decoder_model.get_layer('time1').set_weights(
            self.model.get_layer('time1').get_weights())  
        self.decoder_model.get_layer('time2').set_weights(
            self.model.get_layer('time2').get_weights())  
        
    def generate_data(self, truth, adjacent_truth,
                      val_split_size=0.8, test_split_size=0.8):
        encoder1_inputs = []
        encoder2_inputs = []
        decoder_inputs = []
        y_hat = []
        
        for i in range(len(truth) - self.look_back - self.look_ahead):
            encoder1_inputs.append(truth[i:i+self.look_back])
            encoder2_inputs.append(adjacent_truth[i:i+self.look_back])
            decoder_inputs.append(truth[i+self.look_back-1:i+self.look_back+self.look_ahead-1])
            y_hat.append(truth[i+self.look_back:i+self.look_back+self.look_ahead])
            
        encoder1_inputs = np.array(encoder1_inputs)
        encoder2_inputs = np.array(encoder2_inputs)
        decoder_inputs = np.array(decoder_inputs)
        y_hat = np.array(y_hat)
        
        test_size = int(len(truth) * test_split_size)
        
        x_train = [encoder1_inputs[:test_size], encoder2_inputs[:test_size], decoder_inputs[:test_size]]
        x_test = [encoder1_inputs[test_size:], encoder2_inputs[test_size:], decoder_inputs[test_size:]]
        x_total = [encoder1_inputs, encoder2_inputs, decoder_inputs]
        
        y_train = y_hat[:test_size]
        y_test = y_hat[test_size:]
        
        return [x_train, x_test, x_total], [y_train, y_test, y_hat], test_size 
    
    def predict(self, enc1_inputs, enc2_inputs, dec_inputs):
        assert len(enc1_inputs) == len(dec_inputs), 'not equal length of enc_inputs(%s) and enc_inputs(%s)' % (len(enc_inputs), len(dec_inputs))
        
        prd_values = []
        # input의 길이만큼 반복
        for i in range(len(enc1_inputs)):
            # encoder만 사용해서 encoder의 output과 states 얻기
            enc_out, enc_h, enc_c = self.encoder_model.predict([enc1_inputs[i].reshape(1, self.look_back, 1),
                                                                enc2_inputs[i].reshape(1, self.look_back, self.num_adjacent)])
            dec_in = dec_inputs[i][0].reshape(-1, 1)
            
            tmp = []
            # 이후는 look ahead만큼 연속적으로 예측한다.
            for ahead in range(self.look_ahead):
                # decoder는 받은 첫 번째 값으로 예측 및 states를 반환
                prd, dec_h, dec_c = self.decoder_model.predict([dec_in] + [enc_out, enc_h, enc_c])
                prd_val = prd[0, 0, 0]
                
                if prd_val > 0:
                    tmp.append(prd_val)
                else:
                    # 예측 값이 0보다 작으면 아주 작은 값을 줌
                    # 0일 경우 MAPE, CORR 계산에서 오류가 날 수 있음
                    tmp.append(np.random.uniform(1e-6, 1e-5))
                
                # 연속적인 예측을 위해 다음 decoder input을 예측된 값으로 설정
                dec_in = prd[0, 0]
                # decoder에 들어갈 states를 반환된 states로 설정
                enc_h, enc_c = dec_h, dec_c
            
            prd_values.append(tmp)
        prd_values = np.array(prd_values)
        return prd_values              


'''
E3: 해당 주 확진자 + 시간 차 웹데이터
'''
class E3():
    def __init__(self, **kwargs):
        DENSE_UNIT = 64
        ENC_LSTM_UNIT = 128
        DEC_LSTM_UNIT = ENC_LSTM_UNIT * 2 * 2
        DROPOUT = 0.2
        
        self.look_back = kwargs['look_back']
        self.look_ahead = kwargs['look_ahead']
        self.num_adjacent = kwargs['num_adjacent']
        self.num_words = kwargs['num_words']
        
        ### 학습 과정 ###
        # encoder 1
        enc1_inputs = Input(shape=(self.look_back, 1), name='enc1_input')
        
        # LSTM 2개 사용
        enc1_lstm1 = Bidirectional(LSTM(ENC_LSTM_UNIT, return_state=True, return_sequences=True,
                          dropout=DROPOUT, name='enc1_lstm1'), name='enc1_bilstm1')
        enc1_lstm2 = Bidirectional(LSTM(ENC_LSTM_UNIT, return_state=True, return_sequences=True,
                          dropout=DROPOUT, name='enc1_lstm2'), name='enc1_bilstm2')
        enc1_outputs, _, _, _, _ = enc1_lstm1(enc1_inputs)
        enc1_outputs, enc1_fh, enc1_fc, enc1_bh, enc1_bc = enc1_lstm2(enc1_outputs)
        enc1_h = Concatenate(name='enc1_h')([enc1_fh, enc1_bh])
        enc1_c = Concatenate(name='enc1_c')([enc1_fc, enc1_bc])
        
        # encoder 3
        enc3_inputs = Input(shape=(self.look_back, self.num_words), name='enc3_input')
        enc3_lstm1 = Bidirectional(LSTM(ENC_LSTM_UNIT, return_state=True, return_sequences=True,
                          dropout=DROPOUT, name='enc3_lstm1'), name='enc3_bilstm1')
        enc3_lstm2 = Bidirectional(LSTM(ENC_LSTM_UNIT, return_state=True, return_sequences=True,
                          dropout=DROPOUT, name='enc3_lstm2'), name='enc3_bilstm2')
        enc3_outputs, _, _, _, _ = enc3_lstm1(enc3_inputs)
        enc3_outputs, enc3_fh, enc3_fc, enc3_bh, enc3_bc = enc3_lstm2(enc3_outputs)
        enc3_h = Concatenate(name='enc3_h')([enc3_fh, enc3_bh])
        enc3_c = Concatenate(name='enc3_c')([enc3_fc, enc3_bc])
        
        enc_outputs = Concatenate(axis=-1, name='enc_outputs_concat')([enc1_outputs, enc3_outputs])
        enc_states = [
            Concatenate(name='enc_h')([enc1_h, enc3_h]),
            Concatenate(name='enc_c')([enc1_c, enc3_c]),
        ]
        
        
        # decoder의 input shape. (None, 1)
        dec_inputs = Input(shape=(None, self.num_words+1), name='dec_input')
        dec_lstm1 = LSTM(DEC_LSTM_UNIT, return_state=True, return_sequences=True,
                         dropout=DROPOUT, name='dec_lstm1')
        dec_outputs, _, _ = dec_lstm1(dec_inputs, initial_state=enc_states)
        
        dec_attn = Attention(name='dec_attn')([dec_outputs, enc_outputs])
        dec_concat = Concatenate(name='dec_concat')([dec_outputs, dec_attn])
        
        dec_dense1 = TimeDistributed(Dense(DENSE_UNIT, name='dense1'), name='time1')
        dec_dense2 = TimeDistributed(Dense(1, name='dense2'), name='time2')
        leaky_relu = LeakyReLU(alpha=0.1)
        outputs = dec_dense1(dec_concat)
        outputs = dec_dense2(outputs)
        outputs = leaky_relu(outputs)
        
        model = Model([enc1_inputs, enc3_inputs, dec_inputs], outputs)
        model.compile(optimizer='adam', 
                      loss='mse')
        
        self.model = model
        
        ### 예측 과정 ###
        # 예측 과정에서 decoder는 states를 입력 받아야함
        dec_state_inputs = [
            Input(shape=(DEC_LSTM_UNIT,), name='dec_h_input'),
            Input(shape=(DEC_LSTM_UNIT,), name='dec_c_input')
        ]
        dec_hidden_inputs = Input(shape=(self.look_back, DEC_LSTM_UNIT), name='dec_hidden_input')
        
        dec_outputs2, dec_state_h, dec_state_c = dec_lstm1(dec_inputs,
                                                           initial_state=dec_state_inputs)
        dec_attn2 = Attention(name='dec_test_attn2')([dec_outputs2, dec_hidden_inputs])
        dec_concat2 = Concatenate(name='dec_test_concat2')([dec_outputs2, dec_attn2])
        dec_outputs2 = dec_dense1(dec_concat2)
        dec_outputs2 = dec_dense2(dec_outputs2)
        dec_outputs2 = leaky_relu(dec_outputs2)
        
        self.encoder_model = Model([enc1_inputs, enc3_inputs], [enc_outputs] + enc_states)
        self.decoder_model = Model([dec_inputs] + [dec_hidden_inputs] + dec_state_inputs,
                                   [dec_outputs2] + [dec_state_h, dec_state_c])
        
    def fit(self, x, y, epochs=10, batch_size=32, validation_split=0.2, callbacks=None,
            shuffle=False, verbose=1):
        history = self.model.fit(x, y, epochs=epochs, batch_size=batch_size,
                                validation_split=validation_split, callbacks=callbacks,
                                shuffle=shuffle, verbose=verbose)
        return history
    
    def load_model(self, model_path=None):
        self.model = load_model(model_path)
        
        # model을 load한 후에, 
        # encoder_model, decoder_model 에 weight 설정
        self.encoder_model.get_layer('enc1_bilstm1').set_weights(
            self.model.get_layer('enc1_bilstm1').get_weights())
        self.encoder_model.get_layer('enc1_bilstm2').set_weights(
            self.model.get_layer('enc1_bilstm2').get_weights())
        self.encoder_model.get_layer('enc3_bilstm1').set_weights(
            self.model.get_layer('enc3_bilstm1').get_weights())
        self.encoder_model.get_layer('enc3_bilstm2').set_weights(
            self.model.get_layer('enc3_bilstm2').get_weights())
        
        self.decoder_model.get_layer('dec_lstm1').set_weights(
            self.model.get_layer('dec_lstm1').get_weights())
        
        self.decoder_model.get_layer('time1').set_weights(
            self.model.get_layer('time1').get_weights())  
        self.decoder_model.get_layer('time2').set_weights(
            self.model.get_layer('time2').get_weights())  
        
    def generate_data(self, truth, adjacent_truth, norm_df, lag_df,
                      max_lag=28, val_split_size=0.8, test_split_size=0.8):
        encoder1_inputs = []
        encoder3_inputs = []
        decoder_inputs = []
        y_hat = []
        
        norm_data = []
        norm_df = norm_df.reset_index(drop=True)
        for word in lag_df.columns:
            lag = int(lag_df[word]['max_lag'])
            norm_data.append(norm_df.loc[max_lag-lag:len(norm_df)-1-lag, word].values.reshape(-1, 1))
        norm_data = np.hstack(norm_data)
        
        for i in range(len(truth) - self.look_back - self.look_ahead):
            encoder1_inputs.append(truth[i:i+self.look_back])
            encoder3_inputs.append(norm_data[i:i+self.look_back])
            
            decoder_inputs.append(
                np.hstack(
                    [truth[i+self.look_back-1:i+self.look_back+self.look_ahead-1],
                    norm_data[i+self.look_back-1:i+self.look_back+self.look_ahead-1]])
            )
            y_hat.append(truth[i+self.look_back:i+self.look_back+self.look_ahead])
            
        encoder1_inputs = np.array(encoder1_inputs)
        encoder3_inputs = np.array(encoder3_inputs)
        decoder_inputs = np.array(decoder_inputs)
        y_hat = np.array(y_hat)
        
        test_size = int(len(truth) * test_split_size)
        
        x_train = [encoder1_inputs[:test_size], encoder3_inputs[:test_size], decoder_inputs[:test_size]]
        x_test = [encoder1_inputs[test_size:], encoder3_inputs[test_size:], decoder_inputs[test_size:]]
        x_total = [encoder1_inputs, encoder3_inputs, decoder_inputs]
        
        y_train = y_hat[:test_size]
        y_test = y_hat[test_size:]
        
        return [x_train, x_test, x_total], [y_train, y_test, y_hat], test_size 
    
    def predict(self, enc1_inputs, enc3_inputs, dec_inputs):
        assert len(enc1_inputs) == len(dec_inputs), 'not equal length of enc_inputs(%s) and enc_inputs(%s)' % (len(enc_inputs), len(dec_inputs))
        
        prd_values = []
        # input의 길이만큼 반복
        for i in range(len(enc1_inputs)):
            # encoder만 사용해서 encoder의 output과 states 얻기
            enc_out, enc_h, enc_c = self.encoder_model.predict([enc1_inputs[i].reshape(1, self.look_back, 1),
                                                                enc3_inputs[i].reshape(1, self.look_back, self.num_words)])
            dec_in = dec_inputs[i][0].reshape(1, 1, self.num_words+1)
            
            tmp = []
            # 이후는 look ahead만큼 연속적으로 예측한다.
            for ahead in range(self.look_ahead):
                # decoder는 받은 첫 번째 값으로 예측 및 states를 반환
                prd, dec_h, dec_c = self.decoder_model.predict([dec_in] + [enc_out, enc_h, enc_c])
                prd_val = prd[0, 0, 0]
                
                if prd_val > 0:
                    tmp.append(prd_val)
                else:
                    # 예측 값이 0보다 작으면 아주 작은 값을 줌
                    # 0일 경우 MAPE, CORR 계산에서 오류가 날 수 있음
                    tmp.append(np.random.uniform(1e-6, 1e-5))
                
                dec_in = dec_inputs[i, ahead].reshape(1, 1, self.num_words+1)
                dec_in[0, 0, 0] = prd_val
                # decoder에 들어갈 states를 반환된 states로 설정
                enc_h, enc_c = dec_h, dec_c
            
            prd_values.append(tmp)
        prd_values = np.array(prd_values)
        return prd_values  
    
    
'''
E4: 해당 주 확진자 + 인접 주 확진자 + 시간차 웹데이터
'''
class E4():
    def __init__(self, **kwargs):
        DENSE_UNIT = 64
        ENC_LSTM_UNIT = 128
        DEC_LSTM_UNIT = ENC_LSTM_UNIT * 2 * 3
        DROPOUT = 0.2
        
        self.look_back = kwargs['look_back']
        self.look_ahead = kwargs['look_ahead']
        self.num_adjacent = kwargs['num_adjacent']
        self.num_words = kwargs['num_words']
        
        ### 학습 과정 ###
        # encoder 1
        enc1_inputs = Input(shape=(self.look_back, 1), name='enc1_input')
        
        # LSTM 2개 사용
        enc1_lstm1 = Bidirectional(LSTM(ENC_LSTM_UNIT, return_state=True, return_sequences=True,
                          dropout=DROPOUT, name='enc1_lstm1'), name='enc1_bilstm1')
        enc1_lstm2 = Bidirectional(LSTM(ENC_LSTM_UNIT, return_state=True, return_sequences=True,
                          dropout=DROPOUT, name='enc1_lstm2'), name='enc1_bilstm2')
        enc1_outputs, _, _, _, _ = enc1_lstm1(enc1_inputs)
        enc1_outputs, enc1_fh, enc1_fc, enc1_bh, enc1_bc = enc1_lstm2(enc1_outputs)
        enc1_h = Concatenate(name='enc1_h')([enc1_fh, enc1_bh])
        enc1_c = Concatenate(name='enc1_c')([enc1_fc, enc1_bc])
        
        # encoder 2
        enc2_inputs = Input(shape=(self.look_back, self.num_adjacent), name='enc2_input')
        enc2_lstm1 = Bidirectional(LSTM(ENC_LSTM_UNIT, return_state=True, return_sequences=True,
                          dropout=DROPOUT, name='enc2_lstm1'), name='enc2_bilstm1')
        enc2_lstm2 = Bidirectional(LSTM(ENC_LSTM_UNIT, return_state=True, return_sequences=True,
                          dropout=DROPOUT, name='enc2_lstm2'), name='enc2_bilstm2')
        enc2_outputs, _, _, _, _ = enc2_lstm1(enc2_inputs)
        enc2_outputs, enc2_fh, enc2_fc, enc2_bh, enc2_bc = enc2_lstm2(enc2_outputs)
        enc2_h = Concatenate(name='enc2_h')([enc2_fh, enc2_bh])
        enc2_c = Concatenate(name='enc2_c')([enc2_fc, enc2_bc])
        
        # encoder 3
        enc3_inputs = Input(shape=(self.look_back, self.num_words), name='enc3_input')
        enc3_lstm1 = Bidirectional(LSTM(ENC_LSTM_UNIT, return_state=True, return_sequences=True,
                          dropout=DROPOUT, name='enc3_lstm1'), name='enc3_bilstm1')
        enc3_lstm2 = Bidirectional(LSTM(ENC_LSTM_UNIT, return_state=True, return_sequences=True,
                          dropout=DROPOUT, name='enc3_lstm2'), name='enc3_bilstm2')
        enc3_outputs, _, _, _, _ = enc3_lstm1(enc3_inputs)
        enc3_outputs, enc3_fh, enc3_fc, enc3_bh, enc3_bc = enc3_lstm2(enc3_outputs)
        enc3_h = Concatenate(name='enc3_h')([enc3_fh, enc3_bh])
        enc3_c = Concatenate(name='enc3_c')([enc3_fc, enc3_bc])
        
        enc_outputs = Concatenate(axis=-1, name='enc_outputs_concat')([enc1_outputs, enc2_outputs, enc3_outputs])
        enc_states = [
            Concatenate(name='enc_h')([enc1_h, enc2_h, enc3_h]),
            Concatenate(name='enc_c')([enc1_c, enc2_c, enc3_c]),
        ]
        
        
        # decoder의 input shape. (None, 1)
        dec_inputs = Input(shape=(None, self.num_words+1), name='dec_input')
        dec_lstm1 = LSTM(DEC_LSTM_UNIT, return_state=True, return_sequences=True,
                         dropout=DROPOUT, name='dec_lstm1')
        dec_outputs, _, _ = dec_lstm1(dec_inputs, initial_state=enc_states)
        
        dec_attn = Attention(name='dec_attn')([dec_outputs, enc_outputs])
        dec_concat = Concatenate(name='dec_concat')([dec_outputs, dec_attn])
        
        dec_dense1 = TimeDistributed(Dense(DENSE_UNIT, name='dense1'), name='time1')
        dec_dense2 = TimeDistributed(Dense(1, name='dense2'), name='time2')
        leaky_relu = LeakyReLU(alpha=0.1)
        outputs = dec_dense1(dec_concat)
        outputs = dec_dense2(outputs)
        outputs = leaky_relu(outputs)
        
        model = Model([enc1_inputs, enc2_inputs, enc3_inputs, dec_inputs], outputs)
        model.compile(optimizer='adam', 
                      loss='mse')
        
        self.model = model
        
        ### 예측 과정 ###
        # 예측 과정에서 decoder는 states를 입력 받아야함
        dec_state_inputs = [
            Input(shape=(DEC_LSTM_UNIT,), name='dec_h_input'),
            Input(shape=(DEC_LSTM_UNIT,), name='dec_c_input')
        ]
        dec_hidden_inputs = Input(shape=(self.look_back, DEC_LSTM_UNIT), name='dec_hidden_input')
        
        dec_outputs2, dec_state_h, dec_state_c = dec_lstm1(dec_inputs,
                                                           initial_state=dec_state_inputs)
        dec_attn2 = Attention(name='dec_test_attn2')([dec_outputs2, dec_hidden_inputs])
        dec_concat2 = Concatenate(name='dec_test_concat2')([dec_outputs2, dec_attn2])
        dec_outputs2 = dec_dense1(dec_concat2)
        dec_outputs2 = dec_dense2(dec_outputs2)
        dec_outputs2 = leaky_relu(dec_outputs2)
        
        self.encoder_model = Model([enc1_inputs, enc2_inputs, enc3_inputs], [enc_outputs] + enc_states)
        self.decoder_model = Model([dec_inputs] + [dec_hidden_inputs] + dec_state_inputs,
                                   [dec_outputs2] + [dec_state_h, dec_state_c])
        
    def fit(self, x, y, epochs=10, batch_size=32, validation_split=0.2, callbacks=None,
            shuffle=False, verbose=1):
        history = self.model.fit(x, y, epochs=epochs, batch_size=batch_size,
                                validation_split=validation_split, callbacks=callbacks,
                                shuffle=shuffle, verbose=verbose)
        return history
    
    def load_model(self, model_path=None):
        self.model = load_model(model_path)
        
        # model을 load한 후에, 
        # encoder_model, decoder_model 에 weight 설정
        self.encoder_model.get_layer('enc1_bilstm1').set_weights(
            self.model.get_layer('enc1_bilstm1').get_weights())
        self.encoder_model.get_layer('enc1_bilstm2').set_weights(
            self.model.get_layer('enc1_bilstm2').get_weights())
        self.encoder_model.get_layer('enc2_bilstm1').set_weights(
            self.model.get_layer('enc2_bilstm1').get_weights())
        self.encoder_model.get_layer('enc2_bilstm2').set_weights(
            self.model.get_layer('enc2_bilstm2').get_weights())
        self.encoder_model.get_layer('enc3_bilstm1').set_weights(
            self.model.get_layer('enc3_bilstm1').get_weights())
        self.encoder_model.get_layer('enc3_bilstm2').set_weights(
            self.model.get_layer('enc3_bilstm2').get_weights())
        
        self.decoder_model.get_layer('dec_lstm1').set_weights(
            self.model.get_layer('dec_lstm1').get_weights())
        
        self.decoder_model.get_layer('time1').set_weights(
            self.model.get_layer('time1').get_weights())  
        self.decoder_model.get_layer('time2').set_weights(
            self.model.get_layer('time2').get_weights())  
        
    def generate_data(self, truth, adjacent_truth, norm_df, lag_df,
                      max_lag=28, val_split_size=0.8, test_split_size=0.8):
        encoder1_inputs = []
        encoder2_inputs = []
        encoder3_inputs = []
        decoder_inputs = []
        y_hat = []
        
        norm_data = []
        norm_df = norm_df.reset_index(drop=True)
        for word in lag_df.columns:
            lag = int(lag_df[word]['max_lag'])
            norm_data.append(norm_df.loc[max_lag-lag:len(norm_df)-1-lag, word].values.reshape(-1, 1))
        norm_data = np.hstack(norm_data)
        
        for i in range(len(truth) - self.look_back - self.look_ahead):
            encoder1_inputs.append(truth[i:i+self.look_back])
            encoder2_inputs.append(adjacent_truth[i:i+self.look_back])
            encoder3_inputs.append(norm_data[i:i+self.look_back])
            
            decoder_inputs.append(
                np.hstack(
                    [truth[i+self.look_back-1:i+self.look_back+self.look_ahead-1],
                    norm_data[i+self.look_back-1:i+self.look_back+self.look_ahead-1]])
            )
            y_hat.append(truth[i+self.look_back:i+self.look_back+self.look_ahead])
            
        encoder1_inputs = np.array(encoder1_inputs)
        encoder2_inputs = np.array(encoder2_inputs)
        encoder3_inputs = np.array(encoder3_inputs)
        decoder_inputs = np.array(decoder_inputs)
        y_hat = np.array(y_hat)
        
        test_size = int(len(truth) * test_split_size)
        
        x_train = [encoder1_inputs[:test_size], encoder2_inputs[:test_size], encoder3_inputs[:test_size], decoder_inputs[:test_size]]
        x_test = [encoder1_inputs[test_size:], encoder2_inputs[test_size:], encoder3_inputs[test_size:], decoder_inputs[test_size:]]
        x_total = [encoder1_inputs, encoder2_inputs, encoder3_inputs, decoder_inputs]
        
        y_train = y_hat[:test_size]
        y_test = y_hat[test_size:]
        
        return [x_train, x_test, x_total], [y_train, y_test, y_hat], test_size 
    
    def predict(self, enc1_inputs, enc2_inputs, enc3_inputs, dec_inputs):
        assert len(enc1_inputs) == len(dec_inputs), 'not equal length of enc_inputs(%s) and enc_inputs(%s)' % (len(enc_inputs), len(dec_inputs))
        
        prd_values = []
        # input의 길이만큼 반복
        for i in range(len(enc1_inputs)):
            # encoder만 사용해서 encoder의 output과 states 얻기
            enc_out, enc_h, enc_c = self.encoder_model.predict([enc1_inputs[i].reshape(1, self.look_back, 1),
                                                                enc2_inputs[i].reshape(1, self.look_back, self.num_adjacent),
                                                                enc3_inputs[i].reshape(1, self.look_back, self.num_words)])
            dec_in = dec_inputs[i][0].reshape(1, 1, self.num_words+1)
            
            tmp = []
            # 이후는 look ahead만큼 연속적으로 예측한다.
            for ahead in range(self.look_ahead):
                # decoder는 받은 첫 번째 값으로 예측 및 states를 반환
                prd, dec_h, dec_c = self.decoder_model.predict([dec_in] + [enc_out, enc_h, enc_c])
                prd_val = prd[0, 0, 0]
                
                if prd_val > 0:
                    tmp.append(prd_val)
                else:
                    # 예측 값이 0보다 작으면 아주 작은 값을 줌
                    # 0일 경우 MAPE, CORR 계산에서 오류가 날 수 있음
                    tmp.append(np.random.uniform(1e-6, 1e-5))
                
                dec_in = dec_inputs[i, ahead].reshape(1, 1, self.num_words+1)
                dec_in[0, 0, 0] = prd_val
                # decoder에 들어갈 states를 반환된 states로 설정
                enc_h, enc_c = dec_h, dec_c
            
            prd_values.append(tmp)
        prd_values = np.array(prd_values)
        return prd_values              
    
    
'''
Ensemble: E2+E3 앙상블
'''
class Ensemble():
    def __init__(self, **kwargs):
        
        self.look_ahead = kwargs['look_ahead']
        
        self.INPUT_DIM = 3
        DENSE_UNIT = 64
        
        esb_input = Input(shape=(1, self.INPUT_DIM))
        dense1 = Dense(DENSE_UNIT)
        dense2 = Dense(1)
        
        outputs = dense1(esb_input)
        outputs = dense2(outputs)
        
        model = Model(esb_input, outputs)
        model.compile(optimizer='adam',
                      loss='mse')
        
        self.model = model
        
    def fit(self, x, y, epochs=10, batch_size=32, validation_split=0.2, callbacks=None,
            shuffle=False, verbose=1):
        history = self.model.fit(x, y, epochs=epochs, batch_size=batch_size,
                                validation_split=validation_split, callbacks=callbacks,
                                shuffle=shuffle, verbose=verbose)
        return history
    
    def load_model(self, model_path=None):
        self.model = load_model(model_path)
    
    def generate_data(self, y1_hat, y2_hat, y_truth,
                      test_split_size=0.8):
        for i in range(len(y1_hat)):
            if i == 0:
                x = np.hstack([y1_hat[i].reshape(-1, 1),
                               y2_hat[i].reshape(-1, 1),
                               np.arange(self.look_ahead).reshape(-1, 1)]).reshape(self.look_ahead, 1, self.INPUT_DIM)
                y = y_truth.reshape(-1, 1)[i:i+self.look_ahead]
            else:
                x = np.vstack([x,
                               np.hstack([y1_hat[i].reshape(-1, 1),
                               y2_hat[i].reshape(-1, 1),
                               np.arange(self.look_ahead).reshape(-1, 1)]).reshape(self.look_ahead, 1, self.INPUT_DIM)])
                y = np.vstack([y,
                               y_truth.reshape(-1, 1)[i:i+self.look_ahead]])
                
        test_size = int(len(x) * test_split_size)
        
        view_test_size = int(len(y_truth) * test_split_size)
        
        x_train = x[:test_size]
        x_test = x[test_size:]
        
        y_train = y[:test_size]
        y_test = y[test_size:]
        
        return [x_train, x_test, x], [y_train, y_test, y], view_test_size
    
    def predict(self, x):
        if len(x) == 0:
            x = x.reshape(1, 1, self.INPUT_DIM)
            
        predicted = self.model.predict(x)
        y_hat = [ary.reshape(-1) for ary in np.vsplit(predicted, len(predicted)//self.look_ahead)]
        y_hat = np.array(y_hat)
        
        return y_hat
   

# class basic_s2s():
#     def __init__(self, **kwargs):
#         clear_session()
        
#         ENC_LSTM_UNIT = 128
#         DEC_LSTM_UNIT = 128
#         DROPOUT = 0.2
        
#         self.look_back = kwargs['look_back']
#         self.look_ahead = kwargs['look_ahead']
        
#         encoder_inputs = Input(shape=(None, 1))
#         encoder_lstm1 = LSTM(ENC_LSTM_UNIT, dropout=DROPOUT, 
#                              return_sequences=True, return_state=True)
#         encoder_lstm2 = LSTM(ENC_LSTM_UNIT, dropout=DROPOUT, 
#                              return_sequences=True, return_state=True)
#         encoder_outputs, state_h, state_c = encoder_lstm1(encoder_inputs)
#         encoder_outputs, state_h, state_c = encoder_lstm2(encoder_outputs)
        
#         encoder_states = [state_h, state_c]
        
#         decoder_inputs = Input(shape=(None, 1))
#         decoder_lstm = LSTM(DEC_LSTM_UNIT, dropout=DROPOUT, 
#                             return_sequences=True, return_state=True)
#         decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
#                                              initial_state=encoder_states)
#         decoder_dense = Dense(1)
#         decoder_outputs = decoder_dense(decoder_outputs)
        
#         self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
#         self.model.compile(optimizer='adam', loss='mse')
        
#         #
#         self.encoder_model = Model(encoder_inputs, encoder_states)
        
#         decoder_state_input_h = Input(shape=(DEC_LSTM_UNIT,))
#         decoder_state_input_c = Input(shape=(DEC_LSTM_UNIT,))
#         decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

#         decoder_outputs2, dec_state_h, dec_state_c = decoder_lstm(decoder_inputs,
#                                                                  initial_state=decoder_states_inputs)
#         decoder_states = [dec_state_h, dec_state_c]

#         decoder_outputs2 = decoder_dense(decoder_outputs2)
#         self.decoder_model = Model([decoder_inputs] + decoder_states_inputs,
#                                    [decoder_outputs2] + decoder_states)
        
#     def fit(self, x, y, epochs=10, batch_size=32, validation_split=0.2, callbacks=None,
#             shuffle=False, verbose=1):
#         history = self.model.fit(x, y, epochs=epochs, batch_size=batch_size,
#                                 validation_split=validation_split, callbacks=callbacks,
#                                 shuffle=shuffle, verbose=verbose)
#         return history
    
#     def generate_data(self, truth, test_split_size=0.8):
#         encoder_inputs = []
#         decoder_inputs = []
#         y_hat = []
        
#         for i in range(len(truth) - self.look_back - self.look_ahead):
#             encoder_inputs.append(truth[i:i+self.look_back])
#             decoder_inputs.append(truth[i+self.look_back-1:i+self.look_back+self.look_ahead-1])
#             y_hat.append(truth[i+self.look_back:i+self.look_back+self.look_ahead])
            
#         encoder_inputs = np.array(encoder_inputs)
#         decoder_inputs = np.array(decoder_inputs)
#         y_hat = np.array(y_hat)
        
#         test_size = int(len(truth) * test_split_size)
        
#         x_train = [encoder_inputs[:test_size], decoder_inputs[:test_size]]
#         x_test = [encoder_inputs[test_size:], decoder_inputs[test_size:]]
#         x_total = [encoder_inputs, decoder_inputs]
        
#         y_train = y_hat[:test_size]
#         y_test = y_hat[test_size:]
        
#         return [x_train, x_test, x_total], [y_train, y_test, y_hat], test_size 
    
#     def predict(self, enc_inputs, dec_inputs):
#         assert len(enc_inputs) == len(dec_inputs), 'not equal length of enc_inputs(%s) and enc_inputs(%s)' % (len(enc_inputs), len(dec_inputs))
        
#         prd_values = []
#         # input의 길이만큼 반복
#         for i in range(len(enc_inputs)):
#             states_value = self.encoder_model.predict(enc_inputs[i].reshape(1, self.look_back, 1))
            
#             target_seq = np.zeros((1, 1, 1))
#             target_seq[0, 0, 0] = dec_inputs[i][0, 0]
            
#             tmp = []
#             # 이후는 look ahead만큼 연속적으로 예측한다.
#             for ahead in range(self.look_ahead):
#                 # decoder는 받은 첫 번째 값으로 예측 및 states를 반환
#                 output, h, c = self.decoder_model.predict([target_seq] + states_value)
#                 prd_val = output[0, 0, 0]
                
#                 if prd_val > 0:
#                     tmp.append(prd_val)
#                 else:
#                     # 예측 값이 0보다 작으면 아주 작은 값을 줌
#                     # 0일 경우 MAPE, CORR 계산에서 오류가 날 수 있음
#                     tmp.append(np.random.uniform(1e-6, 1e-5))
                
#                 # 연속적인 예측을 위해 다음 decoder input을 예측된 값으로 설정
#                 target_seq[0, 0, 0] = prd_val
#                 # decoder에 들어갈 states를 반환된 states로 설정
#                 states_value = [h, c]
            
#             prd_values.append(tmp)
#         prd_values = np.array(prd_values)
#         return prd_values     