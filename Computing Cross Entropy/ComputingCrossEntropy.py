import string
import math
import re
from numpy.random import sample
from Problem2 import vocab_play

#open sample text which is our corpus
sampletext = open('sample-text.txt', 'r')


#convert all the data from the sample corpus to lower case letters and remove all extra space 
#and replace it by a single space.Remove the new line characters

sampletext = sampletext.read().lower()
trans = str.maketrans('','', string.punctuation)
sampletext = sampletext.replace('\n','')
sampletext = re.sub( '\s+', ' ', sampletext ).strip()
sampletext = sampletext.translate(trans)


prob_table={}

def get_ngram_value(key,dictionary):
    if not key in dictionary:
        return 1
    return dictionary[key]+1

'''
populate_ngrams(data):For every data passed through the below function,we generate a dictionary of trigrams and bigrams.
trigram_list is a dictionary which holds the count of each trigram we come across and bigram_list is the
dictionary which holds the count of each bigram we come across
The function then returns the dictionaries'''

def populate_ngrams(data):
    n_trigram = 3
    n_bigram = 2
    t_pointer = 0
    b_pointer = 0
    trigram_list={}
    bigram_list={}
    
    while b_pointer < (len(data)):
        bigram_list.update({data[b_pointer:b_pointer+2]:get_ngram_value(data[b_pointer:b_pointer+2],\
                                                                    bigram_list)})
        b_pointer+=1
        
    while t_pointer < (len(data)):
        trigram_list.update({data[t_pointer:t_pointer+3]:get_ngram_value(data[t_pointer:t_pointer+3],\
                                                                    trigram_list)})
        t_pointer+=1
        
    return trigram_list,bigram_list

'''
Based on the trigram passed to the below function we compute the probability of 
(P(c1|c2c3))=count of c2c3c1 in trigram_list/count of c2c3 in bigram_list.Smoothing of 0.1 
is applied to this to make data more precise

(P(c1|c2c3))= C(c2c3c1)+0.1 /(C(c2c3)+ 0.1* V

We then return the probability of the trigram
'''

def get_probablity(c1,c2,c3,trigram_list,bigram_list):
    c123 = c2+c3+c1
    c23 = c2+c3   
    if c123 not in trigram_list.keys():
        probablity = float(0+0.1)/(0.1*37)
    else:
        count_c123 = trigram_list[c123]
        count_c23 = bigram_list[c23]    
        probablity = float(count_c123 + 0.1) / ( count_c23 + (37 * 0.1))
            
    return (c123,probablity)  

'''
for the data sent through the function character_probability we get probability of each of the trigrams and store it in the
dictionary prob_table in the form {trigram:probability}
'''

def character_probability(data):
    character = 2
    while character < len(data):
    #   print (data[character],data[character - 2], data[character - 1])
        key,prob=get_probablity(data[character],data[character - 2], data[character - 1],trigram_list,bigram_list)
        prob_table.update({key:prob})
        character+=1


'''
for the data sent through the function character_probability we get probability of each of the trigrams present in
the data which is stored as prob.
We find another probability p which is the count of trigrams/count of bigrams in the sentence whose entropy we are trying to
find.

The Entropy is now Calculated as -sum of p*log(prob,2)

The functions then returns the entropy of each sentence.

'''

def calculate_entropy(data):
    character = 2
    entropy = 0
    trigram_ex,bigram_ex = populate_ngrams(data)
    while character < len(data):
        p=float(trigram_ex[data[character-2]+ data[character - 1]+ data[character]] + 0.1 )/ (bigram_ex[data[character - 2]+ data[character - 1]] + 0.1*37)      
        key,prob = get_probablity(data[character],data[character - 2], data[character - 1],trigram_list,bigram_list)
        entropy+= float(p * math.log(prob,2))
        character+=1
    
    return - entropy/(len(data)-2)



line1='he somehow made this analogy sound exciting instead of hopeless'
line2='no living humans had skeletal features remotely like these'
line3='frequent internet and social media users do not have higher stress levels'
line4='the sand the two women were sweeping into their dustpans was transferred into plastic bags'


#generate training data from corpus and populate the dictionaries
trigram_list,bigram_list=populate_ngrams(sampletext) #for the corpus
#character_probability(sampletext)
#print (vocab_play)
#Call the calculate_entropy file by passing your sentence as a string to the function as below:
print('Entropy of line1:',calculate_entropy(line1))

print('Entropy of line2:',calculate_entropy(line2))

print('Entropy of line3:',calculate_entropy(line3))

print('Entropy of line4:',calculate_entropy(line4))
