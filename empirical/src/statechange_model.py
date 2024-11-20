import pandas as pd

def initial_books(filename = '../output/databento_collated_depth/glbx-mdp3-20240708.csv'):
    es,mes = open(filename,'r').readlines(2)
    es,mes = map(eval,(es,mes))
    es,mes = map(pd.DataFrame,(es,mes))
    return dict(ES=es,MES=mes)

def read_file(filename = '../output/databento_collated_depth/glbx-mdp3-20240708.csv'):
    df = pd.read_csv(filename,usecols=['instrument','action','side','change','midprice','new_level_size'],sep='|',skiprows=2)
    df.marks = df.marks.apply(eval)
    df.change = df.change.apply(eval)

    return df


#price,midprice,level size

def get_parameters(df):
    params = dict(ES=dict(B=dict(),A=dict()),MES=dict(B=dict(),A=dict()))
    for instrument in ('ES','MES'):
        for side in ('A','B'):
            cancels = df[(df.instrument == instrument) & (df.side==side) & (df.action=='C')]
            params[instrument][side]['cancel_price_sigma'] = np.sqrt(np.mean((cancels.price-cancels.midprice)**2))
            
            cancels = df[(df.instrument == instrument) & (df.side==side) & (df.action=='M')]
            params[instrument][side]['modify_price_sigma'] = np.sqrt(np.mean((cancels.price-cancels.midprice)**2))
            
            cancel_frac = cancels['size']/cancels.new_level_size
            cancel_frac_inverse_sigmoid = np.log(cancel_frac/(1-cancel_frac))
            params[instrument][side]['cancel_frac_inverse_sigmoid_mean'] = cancel_frac_inverse_sigmoid.mean()
            params[instrument][side]['cancel_frac_inverse_sigmoid_std'] = cancel_frac_inverse_sigmoid.std(ddof=0)

            modify_frac = modify['size']/modify.new_level_size
            modify_frac_inverse_sigmoid = np.log(modify_frac/(1-modify_frac))
            params[instrument][side]['modify_frac_inverse_sigmoid_mean'] = modify_frac_inverse_sigmoid.mean()
            params[instrument][side]['modify_frac_inverse_sigmoid_std'] = modify_frac_inverse_sigmoid.std(ddof=0)

            insertions = df[(df.instrument == instrument) & (df.side==side) & (df.action=='A')]
            params[instrument][side]['insertion_price_halfnormal'] = np.sqrt(np.mean((cancels.price-cancels.midprice)**2))

            insertion_size_log = np.log(insertions['size'])
            params[instrument][side]['insertion_size_log_mean'] = insertion_size_log.mean()
            params[instrument][side]['insertion_size_log_std'] = insertion_size_log.std(ddof=0)
    
            trade_size_log = np.log(trade['size'])
            params[instrument][side]['trade_size_log_mean'] = trade_size_log.mean()
            params[instrument][side]['trade_size_log_std'] = trade_size_log.std(ddof=0)

    return params
