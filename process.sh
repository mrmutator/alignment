# requires files:
# raw_training.<l1>
# raw_training.<l2>
# raw_gold.<l1>
# raw_gold.<l2>
# raw_dev.<l1>
# raw_dev.<l2>
# raw_test.<l1>
# raw_test.<l2>

l1=en
l2=fr

python ~/PycharmProjects/alignment/normalize.py raw_training.$l1 &
python ~/PycharmProjects/alignment/normalize.py raw_training.$l2 &
wait

python ~/PycharmProjects/alignment/normalize.py raw_gold.$l1 &
python ~/PycharmProjects/alignment/normalize.py raw_gold.$l2 &
wait

python ~/PycharmProjects/alignment/normalize.py raw_dev.$l1 &
python ~/PycharmProjects/alignment/normalize.py raw_dev.$l2 &
wait

python ~/PycharmProjects/alignment/normalize.py raw_test.$l1 &
python ~/PycharmProjects/alignment/normalize.py raw_test.$l2 &
wait

~/mosesdecoder/scripts/tokenizer/tokenizer.perl -l $l1 < raw_training.$l1.normalized > training.tok.$l1 &
~/mosesdecoder/scripts/tokenizer/tokenizer.perl -l $l2 < raw_training.$l2.normalized > training.tok.$l2 &
wait

~/mosesdecoder/scripts/recaser/train-truecaser.perl --model truecase.model.$l1 --corpus training.tok.$l1 &
~/mosesdecoder/scripts/recaser/train-truecaser.perl --model truecase.model.$l2 --corpus training.tok.$l2 &
wait

~/mosesdecoder/scripts/recaser/truecase.perl --model truecase.model.$l1 < training.tok.$l1  > training.true.$l1 &
~/mosesdecoder/scripts/recaser/truecase.perl --model truecase.model.$l2 < training.tok.$l2  > training.true.$l2 &
wait

~/mosesdecoder/scripts/recaser/truecase.perl --model truecase.model.$l1 < raw_gold.$l1.normalized  > gold.$l1 &
~/mosesdecoder/scripts/recaser/truecase.perl --model truecase.model.$l2 < raw_gold.$l2.normalized  > gold.$l2 &
wait

~/mosesdecoder/scripts/training/clean-corpus-n.perl training.true $l1 $l2 training.clean 1 80
cat gold.$l1 training.clean.$l1 > training.$l1
cat gold.$l2 training.clean.$l2 > training.$l2

~/mosesdecoder/scripts/tokenizer/tokenizer.perl -l $l1 < raw_dev.$l1.normalized > dev.tok.$l1 &
~/mosesdecoder/scripts/tokenizer/tokenizer.perl -l $l2 < raw_dev.$l2.normalized > dev.tok.$l2 &
wait

~/mosesdecoder/scripts/recaser/truecase.perl --model truecase.model.$l1 < dev.tok.$l1  > dev.$l1 &
~/mosesdecoder/scripts/recaser/truecase.perl --model truecase.model.$l2 < dev.tok.$l2  > dev.$l2 &
wait

~/mosesdecoder/scripts/tokenizer/tokenizer.perl -l $l1 < raw_test.$l1.normalized > test.tok.$l1 &
~/mosesdecoder/scripts/tokenizer/tokenizer.perl -l $l2 < raw_test.$l2.normalized > test.tok.$l2 &
wait

~/mosesdecoder/scripts/recaser/truecase.perl --model truecase.model.$l1 < test.tok.$l1  > test.$l1 &
~/mosesdecoder/scripts/recaser/truecase.perl --model truecase.model.$l2 < test.tok.$l2  > test.$l2 &
wait

