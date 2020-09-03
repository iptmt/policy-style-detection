#!/bin/bash
echo -n "Type the NAME and press enter: ";
read;
echo You typed ${REPLY}
tar -cvf ${REPLY}.tar.gz dump/ out/ log/ tmp/
rm -rf dump/*
rm -rf out/*
rm -rf log/*
rm -rf tmp/*
