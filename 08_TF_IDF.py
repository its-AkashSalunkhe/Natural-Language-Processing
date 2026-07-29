import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer



paragraph = """Thank you all so very much. Thank you to the Academy. 
               Thank you to all of you in this room. I have to congratulate 
               the other incredible nominees this year. The Revenant was 
               the product of the tireless efforts of an unbelievable cast
               and crew. First off, to my brother in this endeavor, Mr. Tom 
               Hardy. Tom, your talent on screen can only be surpassed by 
               your friendship off screen … thank you for creating a t
               ranscendent cinematic experience. Thank you to everybody at 
               Fox and New Regency … my entire team. I have to thank 
               everyone from the very onset of my career … To my parents; 
               none of this would be possible without you. And to my 
               friends, I love you dearly; you know who you are. And lastly,
               I just want to say this: Making The Revenant was about
               man's relationship to the natural world. A world that we
               collectively felt in 2015 as the hottest year in recorded
               history. Our production needed to move to the southern
               tip of this planet just to be able to find snow. Climate
               change is real, it is happening right now. It is the most
               urgent threat facing our entire species, and we need to work
               collectively together and stop procrastinating. We need to
               support leaders around the world who do not speak for the 
               big polluters, but who speak for all of humanity, for the
               indigenous people of the world, for the billions and 
               billions of underprivileged people out there who would be
               most affected by this. For our children’s children, and 
               for those people out there whose voices have been drowned
               out by the politics of greed. I thank you all for this 
               amazing award tonight. Let us not take this planet for 
               granted. I do not take tonight for granted. Thank you so very much."""
               



ps = PorterStemmer()
wordnet = WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)

corpus = []

for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]',' ', sentences[i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
    
    
print(corpus)   
    
# Creating the Bag of Words Model

# Also we called as document matrix

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_bow = cv.fit_transform(corpus).toarray()





from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()
x_tf = tf.fit_transform(corpus).toarray()




########################################       REMEMBER       ###########################################

# Notice the table created by Bag of Words there is value 1 that represents perticular thing, (so we get confused that when we say 1, what perticular cell we are talking about from the table)
# However notice the table created by TF-IDF since there is "Log" function we get decimal values, and all values are mostly different (since the values are different we easily get to know what thing we are taling about from the table)


# However - The purpose of TF-IDF is not to make every cell unique.
# Its real purpose is to give more importance to informative words and reduce the importance of very common words.