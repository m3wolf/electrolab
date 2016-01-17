#!/bin/sh

# Pulled from https://github.com/chatcannon/galvani/blob/master/get_testdata.sh

## Test data are posted on FigShare, listed in this article
# http://figshare.com/articles/galvani_test_data/1228760

mkdir -p tests/testdata
cd tests/testdata

BIN=/usr/bin/wget

# https://ndownloader.figshare.com/files/3670071

$BIN https://ndownloader.figshare.com/files/3670071 -O NEI-C10_4cycles_C14.mpt
$BIN https://ndownloader.figshare.com/files/3670065 -O NEI-C10_4cycles_C14.mpr
