#!/bin/bash

TESTDATA="
http://www.cc.gatech.edu/dimacs10/archive/data/delaunay/delaunay_n10.graph.bz2
http://www.cc.gatech.edu/dimacs10/archive/data/random/rgg_n_2_15_s0.graph.bz2
http://www.cc.gatech.edu/dimacs10/archive/data/random/rgg_n_2_16_s0.graph.bz2
http://www.cc.gatech.edu/dimacs10/archive/data/random/rgg_n_2_17_s0.graph.bz2
http://www.cc.gatech.edu/dimacs10/archive/data/random/rgg_n_2_18_s0.graph.bz2
http://www.cc.gatech.edu/dimacs10/archive/data/random/rgg_n_2_19_s0.graph.bz2
http://www.cc.gatech.edu/dimacs10/archive/data/random/rgg_n_2_20_s0.graph.bz2
http://www.cc.gatech.edu/dimacs10/archive/data/random/rgg_n_2_21_s0.graph.bz2
http://www.cc.gatech.edu/dimacs10/archive/data/random/rgg_n_2_22_s0.graph.bz2
http://www.cc.gatech.edu/dimacs10/archive/data/random/rgg_n_2_23_s0.graph.bz2
http://www.cc.gatech.edu/dimacs10/archive/data/random/rgg_n_2_24_s0.graph.bz2
http://www.cc.gatech.edu/dimacs10/archive/data/coauthor/coPapersCiteseer.graph.bz2
http://www.cc.gatech.edu/dimacs10/archive/data/coauthor/coPapersDBLP.graph.bz2
http://www.cc.gatech.edu/dimacs10/archive/data/coauthor/coAuthorsDBLP.graph.bz2
http://www.cc.gatech.edu/dimacs10/archive/data/coauthor/coAuthorsCiteseer.graph.bz2
http://www.cc.gatech.edu/dimacs10/archive/data/coauthor/citationCiteseer.graph.bz2
http://www.cc.gatech.edu/dimacs10/archive/data/kronecker/kron_g500-simple-logn16.graph.bz2
http://www.cc.gatech.edu/dimacs10/archive/data/kronecker/kron_g500-simple-logn17.graph.bz2
http://www.cc.gatech.edu/dimacs10/archive/data/kronecker/kron_g500-simple-logn18.graph.bz2
http://www.cc.gatech.edu/dimacs10/archive/data/kronecker/kron_g500-simple-logn19.graph.bz2
http://www.cc.gatech.edu/dimacs10/archive/data/kronecker/kron_g500-simple-logn20.graph.bz2
http://www.cc.gatech.edu/dimacs10/archive/data/kronecker/kron_g500-simple-logn21.graph.bz2
"

make convert-from-dimacs-main.e



mkdir -p data
cd data

for url in $TESTDATA
do
  base=`echo $url | grep -o "[^/]\+$" | sed "s/\(\.txt\.gz\|\.graph\.bz2\)$//"`
  if [ -f ${base}.bin ]
  then
    echo Skipping $base because it already exists
    continue
  fi
  wget $url
  echo Unzipping and converting
  if [[ $url == *.txt.gz ]]
  then
    gzip -d ${base}.txt.gz
    ../convert-from-snap-main.e ${base}.txt ${base}.bin
    rm ${base}.txt
  else
    bzip2 -d ${base}.graph.bz2
    ../convert-from-dimacs-main.e ${base}.graph ${base}.bin
    rm ${base}.graph
  fi
done


