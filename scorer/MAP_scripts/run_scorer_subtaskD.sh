GOLD=$1
HYP=$2

perl _scripts/resort_subtaskD.pl $HYP $GOLD > $HYP\.resorted || exit -1

python _scorer/ev.py $GOLD $HYP\.resorted > $HYP\.score || exit -2

rm $HYP\.resorted || exit -3
