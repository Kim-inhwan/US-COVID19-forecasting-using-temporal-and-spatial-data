from matplotlib import pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'Times New Roman'})
markers = ['o', '*', 'D', '^', 'x']
colors = ['red', 'blue', 'green', 'violet', 'gray']
line_styles = ['--', '-.', '-', '--', '-.']
alphabets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']

"""
 truth, prd 를 가지고 그래프를 그리는 함수
 truth, prd 의 shape는 evaluate.evaluate의 입력과 같음
 두 가지 그래프를 그리는데,
 첫 번재 그래프는 prd 의 연속적인 예측 값을 그린것
 x=0 일 때, [1, 2, 3, 4, ...]를 그리고 x=1 일 때, [2, 3, 4, 5, ...]를 그림
 두 번재 그래프는 특정 연속 예측 값을 이어서 그린 것
 각 index들의 0번째 값([1, 20, 300, 4000, ..]) 를 그리고
 각 index들의 1번째 값([2, 30, 400, 5000, ...]) 을 그림.
"""


def model_test(truth, prd, look_ahead, val_size=None, test_size=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(truth)
    for i, p in enumerate(prd):
        ax1.plot(range(i, i+look_ahead), p)
        
    ax2.plot(truth)
    for ahead in range(look_ahead):
        ax2.plot(range(ahead, len(prd)+ahead), 
                 [p[ahead] for p in prd], label='ahead:' + str(ahead))
    ax2.legend()
        
    if val_size is not None:
        ax1.axvline(x=val_size, c='k', ls='--')
        ax2.axvline(x=val_size, c='k', ls='--')
    if test_size is not None:
        ax1.axvline(x=test_size, c='k', ls='--')
        ax2.axvline(x=test_size, c='k', ls='--')
    plt.show()  
    
    
def performance_by_ahead(target_states, target_models, performs,
                         save_path=None):
    num_fig = len(target_states)
    fig, axs = plt.subplots(num_fig, 3, figsize=(15, 4*num_fig))

    for i, state in enumerate(target_states):
        for col, ev in enumerate(['rmse', 'mape', 'corr']):
            target_df = performs[[c for c in performs if 'total' in c and ev in c]]
            for j, model_name in enumerate(target_models):
                axs[i, col].plot(target_df.T['%s(%s)' % (state, model_name)].values, marker=markers[j], c=colors[j],
                                 ls=line_styles[j])
                axs[i, col].set_ylabel(ev.upper())
                axs[i, col].set_xticks(np.arange(0, 14, 2))
                axs[i, col].set_xticklabels(np.arange(1, 15, 2))
        axs[i, 1].set_xlabel('(%s) %s' % (alphabets[i], state))        

    plt.legend(loc='upper center', bbox_to_anchor=(-0.75, -0.20), ncol=5, 
               labels=target_models) 
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()    
    
    
def predict_by_ahead(target_states, target_models, look_back, look_ahead, test_size,
                     US_norm_by_states, US_scaler_by_states, prd_by_model,
                     save_path=None):
    for ahead in [1, 7, 14]:
        print('='*30, 'ahead:', ahead, '='*30)
        x_ticks = np.arange(0, 171, 34)

        num_fig = len(target_states)
        fig, axs = plt.subplots(num_fig//3, 3, figsize=(5*3, 4*num_fig//3))

        for i, state in enumerate(target_states):
            row = i // 3
            col = i % 3
            style_idx = 0   
            indexs = US_norm_by_states[state].index
            y_truth = US_norm_by_states[state]['cases'].values[look_back+1:]
            axs[row][col].plot(US_scaler_by_states[state].inverse_transform(y_truth[ahead-1:-look_ahead+ahead-1].reshape(-1, 1)),
                        label='Truth', c='k')
            for model_name in target_models:
                prd = prd_by_model[state][model_name]
                axs[row][col].plot(US_scaler_by_states[state].inverse_transform(prd[:, ahead-1].reshape(-1, 1)),
                           label='%s(%s)' % (state, model_name), c=colors[style_idx], marker=markers[style_idx],
                           markersize=1.5, ls=line_styles[style_idx])
                style_idx += 1
            axs[row][col].set_xlabel('(' + alphabets[i] + ') ' + state)
            axs[row][col].set_ylabel('Confirmed')
            axs[row][col].set_xticks(x_ticks)
            axs[row][col].set_xticklabels([tclbl[5:] for tclbl in US_norm_by_states[state].loc[indexs[look_back+1+ahead-1:len(US_norm_by_states[state])-look_ahead+ahead-1-1], 'date'].values[x_ticks]])
            axs[row][col].axvline(x=test_size-ahead+1, ls=':', c='k')
        plt.legend(loc='upper center', bbox_to_anchor=(-0.75, -0.20), ncol=6,
                   labels=['Truth'] + target_models) 
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    
    