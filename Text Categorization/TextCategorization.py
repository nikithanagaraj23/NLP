import string
import re
import os
import math
from collections import OrderedDict
from itertools import count
from collections import Counter


'''
vocabulary.txt contains the vocabulary set
top_comedy.txt contains the top 20 comedies
top_tragedy.txt contains the top 20 tragedies
comedy_tragedy_likelihood.txt contains the likelihood ratios
play_details.txt contains the classification of each file

'''
vocabulary_file1 = open('vocabulary.txt','w')
top_20_comedies = open('top_comedy.txt','w')
top_20_tragedies = open('top_tragedy.txt','w')
comedy_tragedy_file = open('comedy_tragedy_likelihood.txt','w')
play_details = open('play_details.txt','w')

'''
Global declarations accesible through different functions
'''
vocab_play = {} 
word_play_number = {}
all_comedy={}
all_tragedy={}
vocab_dict = []

comedy_tragedy={}
top_comedy={}
top_tragedy={}
all_text_words = []    
    
    
'''
Get the list of file names that belong to comedies and put into comedyfile_list
Get the list of file names that belong to tragedies and put into tragedyfile_list

'''

path_c = 'shakespeare/comedies' 
files_c = os.listdir(path_c)
comedyfile_list = [i for i in files_c if i.endswith('.txt')]

path_t = 'shakespeare/tragedies' 
files_t = os.listdir(path_t)
tragedyfile_list = [i for i in files_t if i.endswith('.txt')]
    

'''
The below function scraps extra spaces,newlines and makes all the letters lower case
'''


def tokenize_file(file):
    sampletext = open(file, 'r')
    sampletext = sampletext.read().lower()
    trans = str.maketrans('','', string.punctuation)
    sampletext = sampletext.translate(trans)
    sampletext = re.sub( '\s+', ' ', sampletext ).strip()
    sampletext = sampletext.replace('\n',' ')
    sampletext = sampletext.split(' ')
    return (sampletext)

'''
The below helper function gets the value of a key in a dictionary and returns
'''

def get_count(key,dictionary):
    if not key in dictionary:
        return 1
    return dictionary[key]+1



def get_play(key,play,dictionary):
    if not key in dictionary:
        return play + 1
    return dictionary[key]+ play + 1



'''
The below function tokenizes all the words in each of the file sent through the function 
and updates dictionary vocab_dict as {filename:words in file}
'''

def update_dict (type):  
    category_dict={}
    path = 'shakespeare/'+type 
    files = os.listdir(path)
    files_txt = [i for i in files if i.endswith('.txt')]
    for file in files_txt:
        x = tokenize_file('shakespeare/'+type+'/' + file)
        vocab_play.update({file:x})
        all_text_words.extend(x)
        
        

'''
The below function removes the words which appear less than 5 times in all files

'''
                
def generate_vocab_set():
    vocab_set = Counter(all_text_words)
    vocab_set = [w for w in vocab_set.keys() if vocab_set[w] >=5 ]
    return list(vocab_set)

'''
The below function removes words which appear only in one file
'''

    
def remove_single_words (vocab):
    for word in vocab:
        for i in list(vocab_play):
            if word in vocab_play[i]:
                word_play_number.update({word:get_count(word,word_play_number)})

    for i in list(word_play_number):
        if word_play_number[i] < 2:
            word_play_number.pop(i)
            

'''
The below function writes data into files

'''
def write_into_file(occurences,filename):
    for word in list(occurences):
        filename.write(word+'\n')


'''
The below function generates training data by removing the words which are present in the
excluded file (test file)
'''
    
def generate_training_data(file):
    comedy_list=[]
    tragedy_list=[]
    for i in comedyfile_list:
        if i in vocab_play.keys() and not i == file:
            comedy_list.extend(vocab_play[i])
            
    for i in tragedyfile_list:
        if i in vocab_play.keys() and not i == file:
            tragedy_list.extend(vocab_play[i])
            
    return comedy_list,tragedy_list
        


'''
The below function populates a dictionary to include the file name as the key and
 log likelihood feature is the value
'''
 
def log_likelihood(data):
    comedy_list,tragedy_list = generate_training_data('')  
    for feature in data:       
        likelihood_comedy =  float(comedy_list.count(feature)+0.1)\
                                         /(len(comedy_list) + 0.1 * len(word_play_number))
        likelihood_tragedy = float(tragedy_list.count(feature)+0.1)\
                                         /(len(tragedy_list) + 0.1 * len(word_play_number))
                   
        comedy_tragedy.update({feature:math.log2(float (likelihood_comedy)/likelihood_tragedy )})
        
    return comedy_tragedy


'''
The below function computes the probability of each of the file in the leave-one-out fashion
'''             

def find_prob(data,comedy_list,tragedy_list):
    prob_comedy = math.log(0.5,2)
    prob_tragedy = math.log(0.5,2)
      
    word_parsed=[]
    for i in data:
        if i in word_play_number:
            if i not in word_parsed:
            #print(prob_comedy)          
                prob_comedy += math.log((float(comedy_list.count(i))+0.1)\
                                         /(len(comedy_list) + 0.1 * len(word_play_number)),2) * data.count(i)
                prob_tragedy += math.log((float(tragedy_list.count(i)+0.1)\
                                          )/(len(tragedy_list) + 0.1 * len(word_play_number)),2) * data.count(i)
                
                word_parsed.append(i)
                
    return prob_comedy,prob_tragedy

def get_genre(file):
    if file in comedyfile_list:
        return 'Comedy'
    
    if file in tragedyfile_list:
        return 'Tragedy'

printText=''

def get_probability_all_files():
    global printText  
     
    print('Comedy list')
    
    for file1 in comedyfile_list:
        comedy_list,tragedy_list =generate_training_data(file1)
        current_file=vocab_play[file1]
        prob_comedy,prob_tragedy=find_prob(current_file, comedy_list, tragedy_list)
        
        if prob_comedy>prob_tragedy:
            print('FileName:',file1,'  True genre:',get_genre(file1),'  Model Genre:Comedy likelihood Ratio:',\
                  float(prob_comedy-prob_tragedy) )
            printText+='FileName:'+file1+'  True genre:'+get_genre(file1)+'  Model Genre: Comedy  likelihood Ratio:'+\
                  str(float(prob_comedy-prob_tragedy))+'\n'
            
        else:
            print('FileName:',file1,'  True genre:',get_genre(file1),'  Model Genre:Tragedy likelihood Ratio:',\
                  float(prob_comedy-prob_tragedy))
            printText+='FileName:'+file1+'  True genre:'+get_genre(file1)+'  Model Genre: Tragedy  likelihood Ratio:'+\
                  str(float(prob_comedy-prob_tragedy))+'\n'
    
    
        all_comedy.update({file1:float(prob_comedy-prob_tragedy)})
    
    print('Tragedy list')
    for file1 in tragedyfile_list:
        comedy_list,tragedy_list =generate_training_data(file1)
        current_file=vocab_play[file1]
        
        prob_comedy,prob_tragedy=find_prob(current_file, comedy_list, tragedy_list)
        
        if prob_comedy < prob_tragedy:
            print('FileName:',file1,'  True genre:',get_genre(file1),'  Model Genre:Tragedy  likelihood Ratio:',\
                  float(prob_comedy-prob_tragedy))
            printText+='FileName:'+file1+' True genre:'+get_genre(file1)+'  Model Genre:Tragedy  likelihood Ratio:'+\
                  str(float(prob_comedy-prob_tragedy))+'\n'
        else:
            print('FileName:',file1,'  True genre:',get_genre(file1),' Model Genre:Comedy  likelihood Ratio:',\
                  float(prob_comedy-prob_tragedy))
            printText+='FileName:'+file1+'  True genre:'+get_genre(file1)+' Model Genre:Comedy  likelihood Ratio:'+\
                  (str(float(prob_comedy-prob_tragedy)))+'\n'
            
        
        all_tragedy.update({file1:float(prob_comedy-prob_tragedy)})        


    
                

word_dict_comedies = update_dict('comedies')
word_dict_tragedies = update_dict('tragedies')
vocab_dict = generate_vocab_set()
remove_single_words(vocab_dict)
word_play_number = list(word_play_number)
print ('Number of words in Vocab set:', len(word_play_number))

'''
write_into_file(word_play_number,vocabulary_file1) #all words from 22 files

#Calculate the Probability for all the test files in leave-one-out fashion
get_probability_all_files()


all_tragedy= OrderedDict(sorted(all_tragedy.items(), key=lambda t: t[1]))
all_comedy= OrderedDict(sorted(all_comedy.items(), key=lambda t: t[1]))

#get the Comedy file which is most likely tragedy and the Tragedy file which is most likely comedy
k,v=all_tragedy.popitem()
k1,v1=all_comedy.popitem(False)


printText+='\n\n' + ' Comedy Most likely Tragedy: ' + k1+':'+str(v1)+'\n'+\
'Tragedy Most Likely Comedy: ' + k +':'+ str(v)+'\n'

play_details.write(printText)
 
'''

comedy_tragedy = log_likelihood(word_play_number)
ordered_likelihood = OrderedDict(sorted(comedy_tragedy.items(), key=lambda t: t[1], reverse=True))


'''
Write the required data into files as required.
'''

for i in comedy_tragedy:
    comedy_tragedy_file.write(i + ':' + str(comedy_tragedy[i])+'\n')

for i in range(20):
    key,item = ordered_likelihood.popitem(False)
    top_20_comedies.write(key + ':'+ str(item) + '\n')
    
for i in range(20):
    key,item = ordered_likelihood.popitem()
    top_20_tragedies.write(key + ':'+ str(item) + '\n')
    
