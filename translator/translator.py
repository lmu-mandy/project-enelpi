# -*- coding: utf-8 -*-
"""
Luis Garcia
Natural Language Processing 
Dr. Korpusik
12/15/2020

This program can train or evaluate a seq2seq model with or without attention for translating from Spanish to Nahuatl. 
It is based on the Pytorch tutorial: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html 
and Dr. Korpusik's modified codebase.
"""

from __future__ import unicode_literals, print_function, division
import fasttext.util
import fasttext
import numpy as np
import matplotlib.ticker as ticker
import time
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import math
from io import open
import unicodedata
import string
import re
import random
import sys, getopt

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import pandas as pd

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

def load_word_embeddings(lang):
    """
        Extracts the fastext vector for a given language, if lang
        does not match a language in the fasttext list this will 
        raise an error.
    """
    fasttext.util.download_model(lang, if_exists='ignore')
    ft = fasttext.load_model('cc.'+lang+'.300.bin')
    # ft = fasttext.load_model('../pre_trained_embeddings/word_embeddings/cc.'+lang+'.300.bin')
    ft.get_dimension()
    lang_vectors = torch.from_numpy(ft.get_input_matrix())
    return lang_vectors

class Lang:
    def __init__(self, name, embedding=None):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
        self.embedding = embedding

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def split_corpus(path):
    """
        Split method thanks to https://datascience.stackexchange.com/questions/15135/train-test-validation-set-splitting-in-sklearn
        returns a 60% - 20% -20% training-validation-test split
    """
    df = pd.read_csv(path, delimiter="|")
    return np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])


def readLangs(lang1, lang2, corpus, reverse=False,word_embeds=False):
    """
        Extracts translation pairs for Nahuatl and Spanish from the given corpus
        and creates Lang objects for Nahuatl and Spanish with optional word embeddings
    """
    pairs = corpus[['Nahuatl', 'Spanish']].to_numpy()
    # Normalizes words
    pairs = [[normalizeString(y) for y in x] for x in pairs]
    # Reverse pairs, make Lang instances

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        if word_embeds:
            input_lang = Lang(lang2,embedding=(load_word_embeddings(lang2)))
            output_lang = Lang(lang1,embedding=load_word_embeddings(lang1))
        else:
            input_lang = Lang(lang2)
            output_lang = Lang(lang1)    
    else:
        if word_embeds:
            input_lang = Lang(lang1,embedding=load_word_embeddings(lang1))
            output_lang = Lang(lang2,embedding=load_word_embeddings(lang2))
        else:
            input_lang = Lang(lang1)
            output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


# May be incresed to get more samples
MAX_LENGTH = 20

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, corpus, reverse=False,quiet=True,word_embeds=False):
    """
        Given a corpus dataframe, and the names of input and output languages (lang1 and lang2)
        will return Lang objects for for the given languages and will extract the filtered 
        translation pairs from the corpus.
        Optionally the input and output languages may be reversed, information about the languages
        in the corpus may be printed, and fasttext word_embeddings may be included so long as the 
        name of the languages match the fasttext names. 
    """
    input_lang, output_lang, pairs = readLangs(lang1, lang2, corpus, reverse,word_embeds=word_embeds)
    og_length = len(pairs)
    pairs = filterPairs(pairs)
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    if not quiet:
        print("Read %s sentence pairs" % og_length)
        print("Trimmed to %s sentence pairs" % len(pairs))
        print("Counting words...")
        print("Counted words:")
        print(input_lang.name, input_lang.n_words)
        print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs



class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, word_embeds=None):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        if word_embeds is not None:
            self.embedding.weight = nn.Parameter(
                word_embeds)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


"""Prepare the training data."""


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ') if word in lang.word2index]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


"""Evaluation"""


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    """
        Translates the given sentence using the given trained encoder and decoder,
        using a predefined Lang object named input_lang.
        The sentence is made to fit the dimension as the encoder and decoder layers.

        It will return a list of decoded words corresponding to the translation as well
        as an attention output if the encoder and decoder networks have an attention mechanism
    """
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(
            max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            try:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
            except:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, pairs, n=1):
    """Translates n sentences from the given translation pairs at random with a default of 1 sentence"""
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def evaluate_bleu(encoder, decoder, pairs, n=10):
    """Evalueates BLEU score over n translation pairs with a default of 10 pairs."""
    references = []
    predictions = []
    for pair in pairs[:n]:
        references.append(pair[1])
        output_words, _ = evaluate(encoder, decoder, pair[0])
        predictions.append(output_words)
    # smoothing function reference here: https://www.nltk.org/_modules/nltk/translate/bleu_score.html
    score = corpus_bleu(references, predictions,
                        smoothing_function=SmoothingFunction().method3)
    print('BLEU score:', score)
    return score

"""
Model training code
"""

teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH, uses_attn=False):
    """
        Trains the encoder and decoder networks a single epoch for the given tensors. 
        Optimizers for each and a loss functions(criterion) are passed in as well. 
    """
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(
        max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            if uses_attn:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            if uses_attn:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points,name):
    """
    Will plot given points on on a uniform y axis relative to the maximum of the points
    and save the figure to a file with the give name. 
    """
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular 
    tick_int = max(points)/10
    loc = ticker.MultipleLocator(base=tick_int)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    ax.set_title(name +" over time")
    plt.savefig(name)


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=1000, learning_rate=0.01, attn=False):
    """
        Trains the encoder and decoder of a Seq2Seq translator model for n_iters (epochs) using a predefined list 
        of translation pairs named training_pairs. 
        It also plots loss and BLEU scores as it trains the model using a predefined list of translation pairs
        named val_pairs
    """
    start = time.time()
    plot_losses = []
    bleu_plot = []

    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, uses_attn=attn)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

            evaluateRandomly(encoder, decoder, val_pairs)
            bleu_plot.append(evaluate_bleu(encoder, decoder, val_pairs))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses,"Loss")
    showPlot(bleu_plot,"BLEU")
    print("Best Bleu Score:", max(bleu_plot))


def save_translator(encoder, decoder, path1="encoder", path2="decoder"):
    """
        Saves an encoder decoder pair to storage to a given path or to a default.
    """
    torch.save(encoder, path1)
    torch.save(decoder, path2)


def load_translator(encoder_path, decoder_path):
    """
        Loads an encoder and a decoder from storage given file paths for each.
    """
    return torch.load(encoder_path), torch.load(decoder_path)


def evaluate_saved(encoder_path, decoder_path, pairs):
    """
        Evaluates the BLEU score of a saved encoder decoder pair for a given dataframe
    """
    evaluate_bleu(*load_translator(encoder_path, decoder_path), pairs)



def main(argv):
    """
        Command Line Interface implementation. 
        -h-> Help: Prints sample command format
        -b-> Build: Sets the mode to build a translation model
        -t-> Translate: Sets the mode to translation requiring encoder and decoder files
        -a-> Attention: For the build mode specifies whether or not to use a decoder with an attention mechanism
        -s,sentence= -> Sentence: For the translation mode, specifies a sentence to be translated to Nahuatl
        -e,encoder= -> Encoder Path: Specifies a path to load an encoder in translation mode or to save an encoder in build mode
        -d,decoder= -> Decoder Path: Specifies a path to load a decoder in translation mode or to save a decoder in build mode
        -i, iters= -> Iteration number: Specifies how many epochs the model should train in build mode
        -p, printevery= > Print frequency: The frequency at which the model will plot Loss and half the rate it will plot validation BLEU scores
        *note* The iteration number must be at least twice that of print frequecy.
        Examples: 
        Translation: python translator.py -t -e encoders/attn_encoder_5K_11-24 -d decoders/attn_decoder_5K_11-24 -s "Muchas Gracias"
        Build: python translator.py -b -a -e encoders/myencoder -d decoders/mydecoder -i 5000 -p 100

    """
    encoder_path = ''
    decoder_path = ''
    test_sentence = ''
    attention = False
    build_model = False
    translate = False
    iterations =100
    doc_gap = 20
    try: 
        opts, args = getopt.getopt(argv,"habts:e:d:i:p:",["sentence=","encoder=","decoder=","iters=","printevery="])
    except getopt.GetoptError:
        print ('translator.py -t -e <encoderfile> -d <decoderfile> -s <sentence>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print ('translator.py -e <encoderfile> -d <decoderfile> -s <sentence>')
            sys.exit()
        elif opt == "-t":
            translate= True
        elif opt == "-b":
            build_model = True
        elif opt =="-a":
            attention= True
        elif opt in ("-s","--sentence"):
            test_sentence= arg
        elif opt in ("-e","--encoder"):
            encoder_path = arg
        elif opt in ("-d","--decoder"):
            decoder_path = arg
        elif opt in ("-i","--iters"):
            try:
                iterations= int(arg)
            except:
                print("iterations must be integers")
                sys.exit()    
        elif opt in ("-p","--printevery"):
            try:
                doc_gap= int(arg)
            except:
                print("printevery must be an integer less than iterations")
                sys.exit()
    if iterations< 2*doc_gap:
        print ("The number of iterations must be at least twice the print frequency.")
        sys.exit(2)
    if build_model==translate:
        print(build_model,translate)
        print ('translator.py -e <encoderfile> -d <decoderfile> -s <sentence>')
        sys.exit(2)
    elif translate:  
        try:
            translation,_= evaluate(*load_translator(encoder_path,decoder_path),test_sentence)
            print(test_sentence + " translates to: " +' '.join(translation)+"\n")
        except:
            print("There is an error with the given paths:",encoder_path,", ",decoder_path)
            print ('translator.py -e <encoderfile> -d <decoderfile> -s <sentence>')
            sys.exit(2)
    else:
        hidden_size = 300
        encoder = EncoderRNN(input_lang.n_words, hidden_size,
                        word_embeds=input_lang.embedding)
        decoder =  AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device) if attention else DecoderRNN(hidden_size, output_lang.n_words).to(device)  
        trainIters(encoder, decoder, iterations, learning_rate=0.01,
            print_every=doc_gap*2, plot_every=doc_gap, attn=attention)
        evaluateRandomly(encoder, decoder, test_pairs)
        evaluate_bleu(encoder, decoder, test_pairs, n=len(test_pairs))

        if encoder_path!='' and decoder_path!='':
            save_translator(encoder,decoder,path1=encoder_path,path2=decoder_path)

if __name__ == "__main__":
    plt.switch_backend('agg') 
    # device = "cpu"
    training, validation, test = split_corpus("../corpus/parallel_nh-es.csv")
    input_lang, output_lang, pairs = prepareData('nah', 'es', training,reverse=True,word_embeds=True)
    _, _, val_pairs = prepareData('nah', 'es', validation, True)
    _, _, test_pairs = prepareData('nah', 'es', test, True)

    main(sys.argv[1:])