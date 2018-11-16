import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import pandas as pd

df = pd.read_csv("./toxic/train.csv")
training = ' '.join(df[df.toxic==1].comment_text.str.lower().values)

ignore_chars = {'<', '^', '`', '\x93', '\x94', '¢', '£', '¤', '¦', '§', '¨', '©',
               '\xad', '®', '¯', '°', '±', '²', '´', '·', '¸', '½', '¿', 'ß', 'à',
               'á', 'ä', 'å', 'æ', 'ç', 'è', 'ê', 'í', 'ï', 'ñ', 'ó', 'ö', 'ù',
               'ú', 'ü', 'þ', 'ą', 'ć', 'đ', 'ė', 'ě', 'ģ', 'ĥ', 'ħ', 'ı', 'ń',
               'ņ', 'ō', 'œ', 'ś', 'ş', 'š', 'ţ', 'ũ', 'ŵ', 'ŷ', 'ż', 'ƒ', 'ǔ',
               'ȳ', '̇', 'ά', 'ί', 'α', 'γ', 'δ', 'ε', 'η', 'θ', 'ι', 'κ', 'λ',
               'μ', 'ν', 'ο', 'π', 'ρ', 'ς', 'σ', 'τ', 'υ', 'φ', 'ω', 'ό', 'ύ',
               'ώ', 'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'и', 'й', 'к', 'л', 'м',
               'н', 'о', 'п', 'р', 'с', 'т', 'у', 'х', 'ц', 'ч', 'щ', 'ъ', 'ы',
               'ь', 'я', 'љ', 'ּ', 'א', 'ב', 'ו', 'י', 'כ', 'ל', 'מ', 'ا', 'ت',
               'س', 'ط', 'ع', 'ف', 'ك', 'ل', 'م', 'ن', 'و', 'ي', 'چ', 'ڜ', 'ڬ',
               'ڰ', 'ڵ', '\u06dd', '۞', '۬', '۵', '۸', 'ۻ', '۾', 'ݓ', 'ݗ', 'ݜ',
               'ݟ', 'ݡ', 'ݣ', 'ݭ', 'ක', 'ත', 'ඳ', 'ර', 'ව', '්', 'ු', 'ᛏ', 'ᵽ',
               'ḟ', 'ḻ', 'ṃ', 'ṗ', 'ṣ', 'ṯ', '–', '‘', '“', '”', '„', '†', '•',
               '…', '\u2060', '₡', '₨', '₩', '₪', '€', '₭', '₳', '₵', '№', '™',
               'ℳ', '⅞', '←', '↑', '→', '↔', '↨', '⇒', '⇔', '∂', '∆', '∇', '−',
               '√', '∞', '∫', '≈', '≠', '≤', '⊕', '─', '╟', '╢', '╦', '►', '◄',
               '★', '☎', '☏', '☥', '☭', '☺', '☻', '☼', '♠', '♣', '♥', '♦', '♪',
               '♫', '✄', '✉', '✋', '✍', '✎', '✽', '❝', '❞', '➨', '⟲', 'ツ', '妈',
               '学', '影', '惑', '武', '永', '烂', '的', '絡', '者', '臭', '見', '訣', '迷',
               '連', '\ufeff', '．', 'ａ', 'ｃ', 'ｋ', 'ｌ', 'ｍ', 'ｎ', 'ｏ', 'ｔ', 'ｗ',
               '🏼', '👍', '💩', '😂', '😄', '😊'}

training = ''.join([ch if ch not in ignore_chars else '\u2600' for ch in training])

chars = sorted(list(set(training)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = {v: k for k, v in char_to_int.items()}

n_chars = len(training)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

seq_length = 1
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = training[i:i + seq_length]
    seq_out = training[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

X = np.reshape(dataX, (n_patterns, seq_length, 1))
X = X / float(n_vocab)
y = np_utils.to_categorical(dataY)

# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath="./models/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X, y, epochs=10, batch_size=256, callbacks=callbacks_list)
