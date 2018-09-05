#!/bin/sh

cd emd/
sh tf_auctionmatch_compile.sh
cd ..

cd grouping/
sh tf_grouping_compile.sh
cd ..


cd interpolation/
sh tf_interpolate_compile.sh
cd ..

cd sampling/
sh tf_sampling_compile.sh
cd ..