import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
import plotly
import plotly.plotly as py
import plotly.graph_objs as go

plotly.tools.set_credentials_file(username='jkenn4', api_key='0GSQOuD2n73387IE6vDL')


#Trade data read in; Trade data includes: Ticker, Buy date, Buy Price   
#Tickers used in training/testing were scanned for using online technical and fundamental scanners
#Buy dates / Buy Prices manually determined

df = pd.read_csv('All trades.csv')


#Drop non-unique tickers / Shorts

df=df.drop(df.index[[236,291,287,281,282,289,272,279,286,290,7,105,120,131,133,137,145,159,160,162,178,185,194,200,207,272,292,10,15,21,37,40,192 ]])


df2=df.loc[df['Buy/Sell'] == "BUY"]
df2['TradeDate'] = pd.to_datetime(df['TradeDate'])

uniquelist=[]
nonuniqueindex=[]

for i in range(len(df2.Symbol.values)):
    if df2.Symbol.values[i] in uniquelist:
        nonuniqueindex.append(i)
    if df2.Symbol.values[i] not in uniquelist:
        uniquelist.append(df2.Symbol.values[i])

df2=df2.drop(df2.index[nonuniqueindex])
len(df2.index.values)

import matplotlib.pyplot as plt
from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates
import statistics
import datetime

df3=df2.set_index("Symbol",inplace=False)

fourWeekReturns=[]
neutralIndices=[]
posIndices=[]
negIndices=[]
lenarr=0

#Following for loop uses each trade (composed of Ticker, Buy Date, Buy Price) and forms images of 4 month price charts for given 
#   stocks. The price chart's last shown date is the 'buy date'. Images are formed for all inputted tickers. Returns (Classes) in this
#   case are determined to be the returns 4 weeks into the future (after the buy date).

for i in range(len(df2.Symbol.values)):
    buydate=df3.loc[df2.Symbol.values[i]]['TradeDate']
    
    #Determine start/end dates for price chart lookback period
    
    start=buydate-datetime.timedelta(days=120)
    end=buydate+datetime.timedelta(days=28)
    
    b = web.DataReader(df2.Symbol.values[i], 'iex', start, end).reset_index()
    b.date = pd.to_datetime(b.date)
    
    a=b
    a=a.set_index("date",inplace=False)
    
    #Below code if you wanted to buy on the close of the buydate as opposed to the actual set buy price
    #buyprice=a.loc[pd.to_datetime(buydate)]['close'] 
    #fourWeekReturnStock = (b.iloc[b.shape[0]-1]['close']-buyprice)/buyprice
    
    #Actual Buy price
    buyprice=df2.iloc[i]['Price']
    fourWeekReturnStock = (b.iloc[b.shape[0]-1]['close']-buyprice)/buyprice
    
    fourWeekReturns.append(fourWeekReturnStock)
    
    ##Standardize Patterns
    
    closescale=[1]
    for i in range(1,b.shape[0]):
        closescale.append(b.iloc[i]['close']/b.iloc[i-1]['close']-1+closescale[i-1])
    
    b['closescaled']=closescale
    
    b['open']=((b['open']-b['close'])/b['close'])*b['closescaled']+b['closescaled']
    b['low']=((b['low']-b['close'])/b['close'])*b['closescaled']+b['closescaled']
    b['high']=((b['high']-b['close'])/b['close'])*b['closescaled']+b['closescaled']
    
    
    #Prep for graph append
    c=b.index[b['date'] == pd.to_datetime(buydate)].tolist()
    b=b.iloc[0:c[0]+1]
    
    df = b[['date', 'open', 'high', 'low', 'closescaled', 'volume']]
    df["date"] = df["date"].apply(mdates.date2num)
    
    f1 = plt.subplot2grid((6, 4), (1, 0), rowspan=6, colspan=4, axisbg='#07000d')
    candlestick_ohlc(f1, df.values, width=.6, colorup='#53c156', colordown='#ff1717')
    
    plt.axis('off')
 
    plt.xticks([])
    plt.yticks([])
    
    #Multi-Class Case - Saving price charts as images to be used for training/testing
    if fourWeekReturnStock>0.04:
        posIndices.append(i)
        lenposarr=len(posIndices)
        plt.savefig('data2/train/Positive/figure(%d).png' % lenposarr)
    elif fourWeekReturnStock<-0.04:
        negIndices.append(i)
        lennegarr=len(negIndices)
        plt.savefig('data2/train/Negative/figure(%d).png' % lennegarr)
    else:
        neutralIndices.append(i)
        lenneutralarr=len(neutralIndices)
        plt.savefig('data2/train/Neutral/figure(%d).png' % lenneutralarr)
    
    '''
    Binary Case:
    
    if fourWeekReturnStock>0.0:
        posIndices.append(i)
        lenposarr=len(posIndices)
        plt.savefig('data2/train/Positive/figure(%d).png' % lenposarr)
    else:
        negIndices.append(i)
        lennegarr=len(negIndices)
        plt.savefig('data2/train/Negative/figure(%d).png' % lennegarr)
    ''' 

#Below is optimal CNN architecture, determined using hyperas package
    
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

import matplotlib 
import pylab as plt
%matplotlib inline
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

#Code for visualizing layer activations - Learned from Francois Chollet's book, Deep Learning w/Python  

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
