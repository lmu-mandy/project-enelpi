
Luis Garcia
A Nahuatl to Spanish NMT implemented by Luis Garcia as a final project for CMSI 537.
# How To Use
I have included a requirements.txt file to easily download the program's requirements using    this command `pip3 install -r requirements.txt`. Once the requirements are installed, One can use the program with a command line interface specified next.

The following is a description on the implemented command line interface: 
- -h-> Help: Prints sample command format
- -b-> Build: Sets the mode to build a translation model
- -t-> Translate: Sets the mode to translation requiring encoder and decoder files
- -a-> Attention: For the build mode specifies whether or not to use a decoder with an attention mechanism
- -s,sentence= -> Sentence: For the translation mode, specifies a sentence to be translated to Nahuatl
- -e,encoder= -> Encoder Path: Specifies a path to load an encoder in translation mode or to save an encoder in build mode
- -d,decoder= -> Decoder Path: Specifies a path to load a decoder in translation mode or to save a decoder in build mode
- -i, iters= -> Iteration number: Specifies how many epochs the model should train in build mode
- -p, printevery= > Print frequency: The frequency at which the model will plot Loss and half the rate it will plot validation BLEU scores
*note* The iteration number must be at least twice that of print frequecy.
        
## Examples: 
*This assumes a user is in the translator directory*

Translation:

`python translator.py -t -e encoders/attn_encoder_5K_11-24 -d decoders/attn_decoder_5K_11-24 -s "Muchas Gracias"`

Build: 

`python translator.py -b -a -e encoders/myencoder -d decoders/mydecoder -i 5000 -p 100`
## Error Handling
The translator requires a powerful machine to build if word embeddings are enabled, to disable word embeddings, edit line 613 (at the time of writing) by setting `word_embeds=False`.

Running the translator will fail if the MAX_LENGTH variable is not set to the dimensions corresponding to the saved encoder and decoder pair. 

### Disclaimer:
The corpus parallel_nh-es.csv has been created with Christos Christodoulopulus's bible-corpus, and I do not claim any credit for the contents of this corpus.
