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
