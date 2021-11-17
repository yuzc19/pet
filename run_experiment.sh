python cli.py \
  --method pet \
  --pattern_ids $PATTERN_IDS \
  --no_distillation \
  --data_dir ../Seq2Seq-Prompt-Learning/data/k-shot/$TASK/16-$SEED \
  --model_type roberta \
  --model_name_or_path roberta-base \
  --task_name $TASK \
  --output_dir result/$TASK-$SEED \
  --seed $SEED \
  --do_train \
  --do_eval