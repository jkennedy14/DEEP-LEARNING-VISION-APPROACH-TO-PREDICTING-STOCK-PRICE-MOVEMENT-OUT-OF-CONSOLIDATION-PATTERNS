import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates
import pandas_datareader.data as web
import statistics
import datetime
%matplotlib inline

'''
*Trade data read in; Trade data includes: Ticker, Buy date, Buy Price   
*Tickers used in training/testing were scanned for using online technical and fundamental scanners
*Buy dates / Buy Prices manually determined
*Shorts/Duplicate ticker orders dropped in excel 
'''

df = pd.read_csv('All trades.csv')

df_buys=df.loc[df['Buy/Sell'] == 'BUY']
df_buys['TradeDate'] = pd.to_datetime(df_buys['TradeDate'])
df_buys.drop_duplicates(subset ="Symbol", keep = 'first', inplace = True) 

#Determine start/end dates for price chart lookback period
df_buys['start_date']=df_buys['TradeDate'].apply(lambda x: x-datetime.timedelta(days=120))
df_buys['end_date']=df_buys['TradeDate'].apply(lambda x: x+datetime.timedelta(days=28))

four_week_returns=[]
neutral_indices=[]
pos_indices=[]
neg_indices=[]

#Following for loop uses each trade (composed of Ticker, Buy Date, Buy Price) and forms images of 4 month price charts for given 
#   stocks. The price chart's last shown date is the 'buy date'. Images are formed for all inputted tickers. Returns (Classes) in this
#   case are determined to be the returns 4 weeks into the future (after the buy date).

for row in df_buys.itertuples():
    idx,symbol,buy_price, buy_date, start, end=row.Index,row.Symbol,row.Price, row.TradeDate,row.start_date, row.end_date   
    symbol_price_data = web.DataReader(symbol, 'iex', start, end).reset_index()
    symbol_price_data.date = pd.to_datetime(symbol_price_data.date)
    
    #Below code if you wanted to buy on the close of the buydate as opposed to the actual set buy price
    #buy_price=symbol_price_data.loc[pd.to_datetime(buy_date)]['close'] 
    #four_week_return_stock = (symbol_price_data.iloc[-1]['close']-buy_price)/buy_price
    
    #Actual Buy price
    four_week_return_stock = (symbol_price_data.iloc[-1]['close']-buy_price)/buy_price
    four_week_returns.append(four_week_return_stock)
    
    ##Standardize Patterns
    close_std=symbol_price_data[['close']]
    close_std=close_std.apply(lambda x: x/x[0])
    symbol_price_data['close_scaled']=close_std['close']
    
    #shorthand
    spd=symbol_price_data.copy()
    
    spd['open']=((spd['open']-spd['close'])/spd['close'])*spd['close_scaled']+spd['close_scaled']
    spd['low']=((spd['low']-spd['close'])/spd['close'])*spd['close_scaled']+spd['close_scaled']
    spd['high']=((spd['high']-spd['close'])/spd['close'])*spd['close_scaled']+spd['close_scaled']
    
    #Prep for graph append
    graph_indices=spd.index[spd['date'] == buy_date].tolist()
    spd=spd.iloc[0:graph_indices[0]+1]
    
    spd_final = spd[['date', 'open', 'high', 'low', 'closescaled', 'volume']]
    spd_final["date"] = spd_final["date"].apply(mdates.date2num)
    
    f1 = plt.subplot2grid((6, 4), (1, 0), rowspan=6, colspan=4, axisbg='#07000d')
    candlestick_ohlc(f1, df.values, width=.6, colorup='#53c156', colordown='#ff1717')
    
    plt.axis('off')
 
    plt.xticks([])
    plt.yticks([])
    
    #Multi-Class Case - Saving price charts as images to be used for training/testing
    if four_week_return_stock>0.04:
        pos_indices.append(i)
        lenposarr=len(pos_indices)
        plt.savefig('data2/train/Positive/figure(%d).png' % lenposarr)
    elif four_week_return_stock<-0.04:
        neg_indices.append(i)
        lennegarr=len(neg_indices)
        plt.savefig('data2/train/Negative/figure(%d).png' % lennegarr)
    else:
        neutral_indices.append(i)
        lenneutralarr=len(neutral_indices)
        plt.savefig('data2/train/Neutral/figure(%d).png' % lenneutralarr)
    
    '''
    Binary Case:
    
    if four_week_return_stock>0.0:
        pos_indices.append(i)
        lenposarr=len(pos_indices)
        plt.savefig('data2/train/Positive/figure(%d).png' % lenposarr)
    else:
        neg_indices.append(i)
        lennegarr=len(neg_indices)
        plt.savefig('data2/train/Negative/figure(%d).png' % lennegarr)
    ''' 

#Below is optimal CNN architecture, determined using hyperas package (*Separate script)
    
from keras.layers import Dense, Dropout, Activation, BatchNormalization

model=models.Sequential()
model.add(layers.Conv2D(4,(3,3), activation='relu', input_shape=(150,150,3)))
model.add(BatchNormalization())
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(16,(3,3), activation='relu'))
model.add(BatchNormalization())
model.add(layers.MaxPooling2D((2,2)))
    
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(BatchNormalization())
model.add(layers.MaxPooling2D((2,2)))
    
model.add(layers.Conv2D(32,(3,3), activation='relu'))
model.add(BatchNormalization())
model.add(layers.MaxPooling2D((2,2)))
    
model.add(layers.Conv2D(16,(3,3), activation='relu'))
model.add(BatchNormalization())
model.add(layers.MaxPooling2D((2,2)))
       
model.add(layers.Flatten())
model.add(Dropout(0.48))
model.add(layers.Dense(21,activation='relu'))
model.add(layers.Dense(3,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=optimizers.adam(lr=1e-4), metrics=['acc'])
model.summary()

history = model.fit_generator(train_generator, 
                              steps_per_epoch=20, 
                              epochs=20, 
                              validation_data=validation_generator, 
                              validation_steps=20)

#Graphs of validation loss / acc

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

#Code for visualizing layer activations - Based off work from Francois Chollet's book, Deep Learning w/Python  

layer_outputs= [layer.output for layer in model.layers[:8]]
activation_model= models.Model(inputs=model.input, outputs=layer_outputs)

activations=activation_model.predict(x)

first_layer_activation= activations[0]

plt.matshow(first_layer_activation[0,:,:,4], cmap='viridis')
plt.show()

#Visualizing every channel in every intermediate activation

layer_names=[]
for layer in model.layers[:8]:
    layer_names.append(layer.name)
    
images_per_row=16

for layer_name, layer_activation in zip(layer_names, activations):
    n_features=layer_activation.shape[-1]
    
    size=layer_activation.shape[1]
    
    n_cols=n_features // images_per_row
    
    display_grid= np.zeros((size*n_cols, images_per_row*size))
    
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image=layer_activation[0,:,:, col*images_per_row+row]
            
            channel_image-= channel_image.mean()
            channel_image/= channel_image.std()
            channel_image*=64
            channel_image+= 128
            
            channel_image= np.clip(channel_image,0,255).astype('uint8')
            display_grid[col*size : (col+1)*size, row*size : (row+1)*size]= channel_image
            
    scale= 1./size
    plt.figure(figsize=(scale*display_grid.shape[1], scale*display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()

#Activation heat map for given testing image
    
neutral_output=model.output[:,1]
last_conv_layer=model.get_layer('conv2d_23') #last conv layer

from keras import backend as K

grads=K.gradients(neutral_output, last_conv_layer.output)[0]
pooled_grads=K.mean(grads,axis=(0,1,2))

iterate=K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

pooled_grads_value, conv_layer_output_value=iterate([x])

for i in range(128):
    conv_layer_output_value[:,:,i]*= pooled_grads_value[i]
    
heatmap=np.mean(conv_layer_output_value, axis=-1)

heatmap=np.maximum(heatmap,0)
heatmap/=np.max(heatmap)
plt.matshow(heatmap)
plt.show()
