#!/usr/bin/perl
#
#  Author: Preslav Nakov
#  
#  Description: Resorts the output files for subtask D
#
#  Last modified: February 5, 2016
#
#

use warnings;
use strict;
use utf8;

my $PREDICTIONS_FILE = $ARGV[0];
my $RELEVANCY_FILE   = $ARGV[1];

#################
####   MAIN   ###
#################

my %scores = ();

### 1. Fetch the scores from the predictions file
open(PRED, $PREDICTIONS_FILE) or die;
binmode(PRED, ":utf8");
while (<PRED>) {
   s/[\n\r]+$//;
   # 201399   7480  16 0.0625   true
   die "Wrong file format!" if (!/^([0-9]+)\t+([0-9]+)\t([0-9\.\-\+eE]+)\t([0-9\.\-\+eE]+)\t(true|false)$/);
   my ($qid, $qaid, $rank, $score, $label) = ($1, $2, $3, $4, $5);
   $scores{"$qid, $qaid"}->{RANK}  = $rank;
   $scores{"$qid, $qaid"}->{SCORE} = $score;
   $scores{"$qid, $qaid"}->{LABEL} = $label;
}
close(PRED) or die;

### 2. Rewrite the predictions
open (INPUT, $RELEVANCY_FILE) or die;
binmode(INPUT, ":utf8");

while (<INPUT>) {
   s/[\n\r]+$//;

   # 201399   7480  16 0.0625   true
   die "Wrong file format!" if (!/^([0-9]+)\t([0-9]+)\t([0-9]+)\t([0-9\.\-eE]+)\t(true|false)$/);
   my ($qid, $qaid, $rank, $score, $label) = ($1, $2, $3, $4, $5);

   die "ERROR: missing entry: $qid, $qaid" if (!defined($scores{"$qid, $qaid"}));

   print "$qid\t$qaid\t", $scores{"$qid, $qaid"}->{RANK}, "\t", $scores{"$qid, $qaid"}->{SCORE}, "\t", $scores{"$qid, $qaid"}->{LABEL}, "\n";
}

close(INPUT) or die;
