## Steps to Run:
1. Put `data_dir` under the same directory as `GPT-2`
2. Run the command to train:
   ```
   $ python train.sh $MODEL_DIR $ACC_STEP $EPOCH
   ```
3. After the training is done, run the command to generate the results:
   ```
   $ python generate.sh $MODEL_DIR
   ```
