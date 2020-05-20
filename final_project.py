from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext("local[*]")
spark = SparkSession(sc)

df_b = spark.read.csv( "yelp-dataset/yelp_business.csv", header=True)
df_r = spark.read.csv( "yelp-dataset/yelp_review.csv", header=True)
df_a = spark.read.csv( "yelp-dataset/yelp_business_attributes.csv", header=True)


from pyspark.sql.functions import lower,col
#spbus_df=df_b.filter(lower(df_b["categories"]).like("%sport%")|lower(df_b["name"]).like("%sport%"))
spbus_df=df_b.filter(lower(df_b["categories"]).like("%sporting goods%"))
###### REMOVE limit(1000) to run in the cluster
sprev_df = df_r.join(spbus_df.select("business_id"), ["business_id", "business_id"]) #.limit(1000)
spatt_df = df_a.join(spbus_df, ["business_id", "business_id"])

from pyspark.sql.functions import udf, regexp_replace
from pyspark.sql.types import ArrayType, StringType

#Reviews
sprev_df = sprev_df.withColumnRenamed("stars", "label")
sprev_df = sprev_df.withColumn("label", sprev_df["label"].cast("double"))
sprev_df = sprev_df.where(col("label").isNotNull())
sprev_df = sprev_df.na.fill("0",["funny", "cool","useful"])
sprev_df = sprev_df.select("text", "label")
# Clean text
sprev_df = sprev_df.withColumn("text", lower(regexp_replace(sprev_df["text"],"[^a-zA-Z\\s]","")))

#Business
spbus_df = spbus_df.dropna(subset=["latitude","longitude"])
#Attributes
spatt_df = spatt_df.na.fill("0")

#Clean and Pre-processing
from pyspark.ml.feature import  StopWordsRemover, CountVectorizer,Tokenizer
from nltk.stem.snowball import SnowballStemmer


#Tokenize the text in the text column
tokenizer = Tokenizer(inputCol="text", outputCol="words")
wordsDataFrame = tokenizer.transform(sprev_df)


#remove 20 most occuring documents, documents with non numeric characters, and documents with <= 3 characters
cv_tmp = CountVectorizer(inputCol="words", outputCol="tmp_vectors")
cv_tmp_model = cv_tmp.fit(wordsDataFrame)


top20 = list(cv_tmp_model.vocabulary[0:20])
more_then_3_charachters = [word for word in cv_tmp_model.vocabulary if len(word) <= 3]
contains_digits = [word for word in cv_tmp_model.vocabulary if any(char.isdigit() for char in word)]
englishwords = StopWordsRemover.loadDefaultStopWords("english")
stopwords = ["a","able","about","above","abst","accordance","according","accordingly","across","act","actually",
             "added","adj","affected","affecting","affects","after","afterwards","again","against","ah","all",
             "almost","alone","along","already","also","although","always","am","among","amongst","an","and",
             "announce","another","any","anybody","anyhow","anymore","anyone","anything","anyway","anyways",
             "anywhere","apparently","approximately","are","aren","arent","arise","around","as","aside","ask",
             "asking","at","auth","available","away","awfully","b","back","be","became","because","become",
             "becomes","becoming","been","before","beforehand","begin","beginning","beginnings","begins",
             "behind","being","believe","below","beside","besides","between","beyond","biol","both","brief",
             "briefly","but","by","c","ca","came","can","cannot","cant","cause","causes","certain","certainly",
             "co","com","come","comes","contain","containing","contains","could","couldnt","d","date","did",
             "didnt","different","do","does","doesnt","doing","done","dont","down","downwards","due","during",
             "e","each","ed","edu","effect","eg","eight","eighty","either","else","elsewhere","end","ending",
             "enough","especially","et","et-al","etc","even","ever","every","everybody","everyone","everything",
             "everywhere","ex","except","f","far","few","ff","fifth","first","five","fix","followed","following",
             "follows","for","former","formerly","forth","found","four","from","further","furthermore","g","gave",
             "get","gets","getting","give","given","gives","giving","go","goes","gone","got","gotten","h","had",
             "happens","hardly","has","hasnt","have","havent","having","he","hed","hence","her","here",
             "hereafter","hereby","herein","heres","hereupon","hers","herself","hes","hi","hid","him","himself",
             "his","hither","home","how","howbeit","however","hundred","i","id","ie","if","ill","im","immediate",
             "immediately","importance","important","in","inc","indeed","index","information","instead","into",
             "invention","inward","is","isnt","it","itd","itll","its","itself","ive","j","just","k","keep",
             "keeps","kept","kg","km","know","known","knows","l","largely","last","lately","later","latter",
             "latterly","least","less","lest","let","lets","like","liked","likely","line","little","ll","look",
             "looking","looks","ltd","m","made","mainly","make","makes","many","may","maybe","me","mean","means",
             "meantime","meanwhile","merely","mg","might","million","miss","ml","more","moreover","most","mostly",
             "mr","mrs","much","mug","must","my","myself","n","na","name","namely","nay","nd","near","nearly",
             "necessarily","necessary","need","needs","neither","never","nevertheless","new","next","nine",
             "ninety","no","nobody","non","none","nonetheless","noone","nor","normally","nos","not","noted",
             "nothing","now","nowhere","o","obtain","obtained","obviously","of","off","often","oh","ok","okay",
             "old","omitted","on","once","one","ones","only","onto","or","ord","other","others","otherwise",
             "ought","our","ours","ourselves","out","outside","over","overall","owing","own","p","page","pages",
             "part","particular","particularly","past","per","perhaps","placed","please","plus","poorly",
             "possible","possibly","potentially","pp","predominantly","present","previously","primarily",
             "probably","promptly","proud","provides","put","q","que","quickly","quite","qv","r","ran","rather",
             "rd","re","readily","really","recent","recently","ref","refs","regarding","regardless","regards",
             "related","relatively","research","respectively","resulted","resulting","results","right","run","s",
             "said","same","saw","say","saying","says","sec","section","see","seeing","seem","seemed","seeming",
             "seems","seen","self","selves","sent","seven","several","shall","she","shed","shell","shes","should",
             "shouldnt","show","showed","shown","showns","shows","significant","significantly","similar","similarly",
             "since","six","slightly","so","some","somebody","somehow","someone","somethan","something","sometime",
             "sometimes","somewhat","somewhere","soon","sorry","specifically","specified","specify","specifying",
             "still","stop","strongly","sub","substantially","successfully","such","sufficiently","suggest","sup",
             "sure","ts","take","taken","tell","tends","th","than","thank","thanks","thanx","that","thats",
             "the","their","theirs","them","themselves","then","thence","there","theres","thereafter","thereby",
             "therefore","therein","theres","thereupon","these","they","theyd","theyll","theyre","theyve","think",
             "third","this","thorough","thoroughly","those","though","three","through","throughout","thru","thus",
             "to","together","too","took","toward","towards","tried","tries","truly","try","trying","twice","two",
             "un","under","unfortunately","unless","unlikely","until","unto","up","upon","us","use","used","useful",
             "uses","using","usually","value","various","very","via","viz","vs","want","wants","was","wasnt","way",
             "we","wed","well","were","weve","welcome","well","went","were","werent","what","whats","whatever",
             "when","whence","whenever","where","wheres","whereafter","whereas","whereby","wherein","whereupon",
             "wherever","whether","which","while","whither","who","whos","whoever","whole","whom","whose","why","will",
             "willing","wish","with","within","without","wont","wonder","would","wouldnt","yes","yet","you","youd",
             "youll","youre","youve","your","yours","yourself","yourselves","zero","template","align","center","cite",
             "style","left","web","flagicon","references","rowspan","width","infobox","colspan","class","time",
             "external","links","background","reflist","area","bgcolor","wikitable","top","http","reflisttemplate",
             "including","list","main","icon","coord","title","webtemplate","border","website","row","include",
             "site","col","set","large","san","valign","color","sort","halign","align","dfdddf","dfbbbf",
             "persondatadefaultsort","rt","really","tried","told","tell","living","getting","pros","cons","come","else",
             "reason","fun","sure","excess","side","way","away","past","cannot","also","even","get","although","nothing",
             "part","confirmed","huge","heard","enjoyed","enjoy","great","excellent","nice","amazing","loved","love",
             "beautiful","awesome", "wonderful","great","bit","thank","fantastic","perfect","little","helpful","happy",
             "highly","incredible","good","terrible","worst","horrible","poor","broken","manager","ever", "worst",
             "awful","hair","dime","family","everyone","easy","quick","treat","terribly","lot","rude","called",
             "car","complained","someone","phone","wynn","except","able","run","nickel","someone","easy","happy",
             "newly","comfimed","theyr","excuse","best","complain","beyond","apology","guest","said","glad",
             "multiple","disgust","friendly","call","bad","wall","charged","sheet","nasty","body","thin","ruined",
             "joke","worse","outstanding","pleasant","promised","elsewhere","terribly","professional",
             "unprofessional","manager","dont","waited","customer","booked","move","disappointed","pm","asked",
             "wrong","money","left","care","bug","changed","change","completely","contacted","work","sit","hour",
             "upgraded","pay","paid","well","old","large","used","sorry","understand","type","smaller","kind","quite",
             "bringing","guess","real","finally","okay","personality","probably","insanely","typically","kinda","yesterday",
             "usually","greatly","need","absolutely","willing","husband","years","itll","yetwel"]

#Combine the four stopwords
stopwords = stopwords + top20 + more_then_3_charachters + contains_digits + englishwords

#Remove stopwords from the tokenized list
remover = StopWordsRemover(inputCol="words", outputCol="filtered", stopWords = stopwords)
wordsDataFrame = remover.transform(wordsDataFrame)

# Stem text
stemmer = SnowballStemmer(language='english')
stemmer_udf = udf(lambda tokens: [stemmer.stem(token) for token in tokens], ArrayType(StringType()))
wordsDataFrame = wordsDataFrame.withColumn("words_stemmed", stemmer_udf("filtered")).select('*', 'words_stemmed')


#Create a new CountVectorizer model without the stopwords
cv = CountVectorizer(inputCol="words_stemmed", outputCol="features")
cvmodel = cv.fit(wordsDataFrame)
df_final_reviews = cvmodel.transform(wordsDataFrame)

### Create the training and testing data
from pyspark.sql.functions import lit
highsat =df_final_reviews.filter(df_final_reviews["label"]>=4).withColumn("label",lit(1))
lowsat = df_final_reviews.filter(df_final_reviews["label"]<3).withColumn("label",lit(0))
highsat_train, highsat_test = highsat.randomSplit([0.7, 0.3], seed=12345)
lowsat_train, lowsat_test = lowsat.randomSplit([0.7, 0.3], seed=12345)
train_data = highsat_train.union(lowsat_train).select("label","features","words_stemmed")
test_data = highsat_test.union(lowsat_test).select("label","features","words_stemmed")


### Logistic Regression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

lr = LogisticRegression(maxIter=20)#, regParam=0.3, elasticNetParam=0)
# Create ParamGrid for Cross Validation
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.0001, 0.0003, 0.0005]) # regularization parameter
             .addGrid(lr.elasticNetParam, [0.0, 0.01, 0.02]) # Elastic Net Parameter (Ridge = 0)
             .build())
# Create 10-fold CrossValidator
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")

cv = CrossValidator(estimator=lr, \
                    estimatorParamMaps=paramGrid, \
                    evaluator=evaluator, \
                    numFolds=10)
cvModel = cv.fit(train_data)
#model = lr.fit(train_data)
predictions = cvModel.transform(test_data)
# Evaluate best model
e = evaluator.evaluate(predictions)
print ("################ EVALUATION ##############")
print(e)
print("#####################################")
#Grab the positive and negative sentiment reviews
pos_rev = predictions.filter(predictions["prediction"]==1).select("features","words_stemmed")
neg_rev = predictions.filter(predictions["prediction"]==0).select("features","words_stemmed")


### LDA Classification
from pyspark.ml.clustering import LDA

lda = LDA(k=10, maxIter=10)
model_pos = lda.fit(pos_rev)
model_neg = lda.fit(neg_rev)



#Positive Topics
topics = model_pos.describeTopics(10)
print (" ################# POSITIVE TOPICS ################")
for topic in topics.rdd.toLocalIterator():
    print("Topic: " + str(topic[0]))
    words = ""
    for n in range(len(topic[1])):
        words = words + cvmodel.vocabulary[topic[1][n]]+ ","
    print (words)
    
    
#Negative Topics
topics = model_neg.describeTopics(10)
print ("############ NEGATIVE TOPICS ###########")
for topic in topics.rdd.toLocalIterator():
    print("Topic: " + str(topic[0]))
    words = ""
    for n in range(len(topic[1])):
        words = words + cvmodel.vocabulary[topic[1][n]]+ ","
    print (words)
    
#Negative talked about words
######Negative talked about words
print ("############ NEGATIVE WORDS ###########")
neg_freq=neg_rev.select("words_stemmed")
neg_words=neg_freq.rdd.flatMap(lambda a: [(w,1) for w in a.words_stemmed]).reduceByKey(lambda a,b: a+b).sortBy(lambda a:-a[1]).take(20)
print (neg_words)



