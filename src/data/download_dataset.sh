FILE1="data/external/un-general-debates.csv"
if [ -f $FILE1 ]; then
   echo "$FILE1 exists already."
else
   kaggle datasets download -d unitednations/un-general-debates -p data/external/ --unzip
fi

FILE2="data/external/wikipedia-iso-country-codes.csv"
if [ -f $FILE2 ]; then
   echo "$FILE2 exists already."
else
   kaggle datasets download -d juanumusic/countries-iso-codes -p data/external/ --unzip
fi

FILE3="data/external/enwiki_20180420_100d.pkl"
if [ -f $FILE3 ]; then
   echo "$FILE3 exists already."
else
   curl -o data/external/enwiki_20180420_100d.pkl.bz2 http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_100d.pkl.bz2
   bunzip2 data/external/enwiki_20180420_100d.pkl.bz2
fi

