cd ./data

# 下载 TProcess.NLTKSentTokenizer 数据
if [ ! -d "TProcess.NLTKSentTokenizer" ]; then
  mkdir TProcess.NLTKSentTokenizer
  cd TProcess.NLTKSentTokenizer
  wget https://data.thunlp.org/TAADToolbox/punkt.english.pickle.zip --no-check-certificate
  unzip ./punkt.english.pickle.zip
  rm -r ./punkt.english.pickle.zip
  cd ..
else
  echo "TProcess.NLTKSentTokenizer already exists, skipping download."
fi

# 下载 TProcess.NLTKPerceptronPosTagger 数据
if [ ! -d "TProcess.NLTKPerceptronPosTagger" ]; then
  mkdir TProcess.NLTKPerceptronPosTagger
  cd TProcess.NLTKPerceptronPosTagger
  wget https://data.thunlp.org/TAADToolbox/averaged_perceptron_tagger.pickle.zip --no-check-certificate
  unzip ./averaged_perceptron_tagger.pickle.zip
  rm -r ./averaged_perceptron_tagger.pickle.zip
  cd ..
else
  echo "TProcess.NLTKPerceptronPosTagger already exists, skipping download."
fi

# 下载 TProcess.StanfordParser 数据
if [ ! -d "TProcess.StanfordParser" ]; then
  mkdir TProcess.StanfordParser
  cd TProcess.StanfordParser
  wget https://data.thunlp.org/TAADToolbox/stanford_parser_small.zip --no-check-certificate
  unzip ./stanford_parser_small.zip
  rm -r ./stanford_parser_small.zip
  cd ..
else
  echo "TProcess.StanfordParser already exists, skipping download."
fi

# 下载 AttackAssist.SCPN 数据
if [ ! -d "AttackAssist.SCPN" ]; then
  mkdir AttackAssist.SCPN
  cd AttackAssist.SCPN
  wget https://data.thunlp.org/TAADToolbox/scpn.zip --no-check-certificate
  unzip ./scpn.zip
  rm -r ./scpn.zip
  cd ..
else
  echo "AttackAssist.SCPN already exists, skipping download."
fi

cd ..
