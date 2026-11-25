# Assignment 2: Sequence-to-Sequence Learning Using LSTM

## What I tried

This assignment uses an encoder-decoder LSTM to translate English into French. I trained the baseline model on a 12,000-pair sample from the dataset, which contains 175,621 pairs. The sample was split into 9,600 training pairs, 1,200 validation pairs, and 1,200 test pairs.

The baseline used 128-dimensional embeddings, a 256-unit LSTM hidden state, batch size 64, Adam with learning rate 0.001, cross-entropy loss that ignores padding, teacher forcing at 0.5, and 10 epochs. It took about 431.6 seconds on the CPU and had 4,083,836 trainable parameters.

I also ran all nine requested comparisons on the same 12,000-pair sample and split. Each comparison ran for two epochs to keep the CPU experiment manageable. These short runs are useful for comparing early results, but they are not enough to establish which model would perform best after full training.

## Q1 — Preprocessing

I cleaned and tokenized the English and French sentences and used a separate vocabulary for each language. English and French have different words and token-to-number mappings. The vocabularies had 4,566 English tokens and 7,036 French tokens.

The special tokens were:

- `<SOS>` marks the start of a sentence.
- `<EOS>` marks the end of a sentence.
- `<PAD>` makes sentences in a batch the same length.
- `<UNK>` represents a word that is not in the vocabulary.

The maximum sequence length was 20, including `<SOS>` and `<EOS>`. At this limit, 18 English and 42 French training sentences were truncated. Truncation can remove useful context, so a sentence may lose important information.

## Q2–Q3 — Encoder and decoder

The encoder embeds the English tokens and processes them with an LSTM. Its final hidden state and cell state are passed to the French decoder so that generation is conditioned on the input sentence. The decoder uses a separate French embedding and an output layer that produces a score for each French vocabulary token. The model had 4,083,836 trainable parameters.

The hidden state carries the LSTM's current representation, while the cell state provides a gated memory path. LSTM gates can help preserve useful information over multiple steps compared with a basic RNN. The decoder produces one token at a time until it generates `<EOS>` or reaches the maximum length.

This model does not use attention, so it relies on the encoder's final states rather than looking back at each input token during decoding. Compressing a full sentence into those states can make longer or more complex inputs harder to translate.

The decoder uses a separate French vocabulary because its output tokens are French, with their own vocabulary IDs. The encoder's final hidden and cell states give the decoder information about the English input. At every step, the fully connected output layer turns the decoder state into scores over the French vocabulary; greedy decoding chooses the highest-scoring token.

## Q4 — Training

Teacher forcing with ratio 0.5 was used for the baseline. During training, the correct previous French token is supplied to the decoder part of the time. This gives it a useful context and can help learning. At inference, the model has to use its own previous predictions instead, which can cause errors to build up; this difference is called exposure bias.

Padding tokens were ignored in the loss so they would not count as real French targets. Training and validation loss both went down during the 10 epochs. At epoch 10, training loss was 3.6276 and validation loss was 5.1018. The gap suggests the model fits the training data better than unseen data, although this short run does not establish how much additional training would help.

## Q5 — Greedy decoding

The decoder generated translations by choosing the highest-scoring token at each step. This is quick, but each choice is made without checking whether a different sequence would work better. Errors in one word can affect the words that follow.

The saved held-out examples show missing words, incorrect word choices, repetition, and ungrammatical output. Longer sentences often lost more of their meaning, but short sentences were not always translated correctly either. Input words outside the English vocabulary map to `<UNK>`.

The ten held-out examples are saved in [`outputs/test_translation_examples.csv`](outputs/test_translation_examples.csv). Twenty drafted example sentences and their predictions are in [`outputs/original_sentence_evaluation.csv`](outputs/original_sentence_evaluation.csv). A separate set of 20 new prompts is in [`outputs/custom_sentence_evaluation.csv`](outputs/custom_sentence_evaluation.csv).

## Q6–Q8 — Requested comparisons

The assignment asks for hidden dimensions 128, 256, and 512; teacher-forcing ratios 0.0, 0.5, and 1.0; and maximum lengths 10, 20, and 30. All nine settings were run for two epochs on the same train/validation/test split. The full results are saved in [`outputs/seq2seq_experiment_comparison.csv`](outputs/seq2seq_experiment_comparison.csv), and the per-epoch loss history is in [`outputs/seq2seq_experiment_histories.csv`](outputs/seq2seq_experiment_histories.csv).

| Comparison | Setting | Train loss | Validation loss | Test loss | Sample BLEU | Training time | Parameters |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hidden size | 128 | 5.5883 | **5.8151** | **5.8260** | 0.0030 | 63.81 s | 2,656,892 |
| Hidden size | 256 | 5.6081 | 5.8809 | 5.8862 | 0.0030 | 96.46 s | 4,083,836 |
| Hidden size | 512 | 5.6151 | 5.9326 | 5.9345 | 0.0018 | 209.74 s | 7,724,156 |
| Teacher forcing | 0.0 | **5.5555** | **5.8423** | **5.8514** | 0.0017 | 97.72 s | 4,083,836 |
| Teacher forcing | 0.5 | 5.6137 | 5.8947 | 5.8996 | 0.0019 | 96.03 s | 4,083,836 |
| Teacher forcing | 1.0 | 6.4526 | 6.6925 | 6.6974 | **0.0074** | 99.85 s | 4,083,836 |
| Maximum length | 10 | **5.1900** | **5.4856** | **5.4989** | **0.0037** | **38.57 s** | 4,083,836 |
| Maximum length | 20 | 5.5930 | 5.8692 | 5.8730 | 0.0018 | 80.42 s | 4,083,836 |
| Maximum length | 30 | 5.6562 | 5.9485 | 5.9559 | 0.0018 | 158.39 s | 4,083,836 |

Each comparison ran for two epochs; BLEU was calculated on a quick sample of up to 300 test pairs. The length-10 run had the lowest validation loss in these short comparisons, but it also truncated 1,429 English and 2,028 French training sentences; length 20 truncated 18 and 42, while length 30 truncated none. So the better short-run loss at length 10 does not mean it preserves the most translation information. Among the hidden sizes, 128 had the lowest validation loss and used the fewest parameters. Teacher forcing 0.0 had the lowest train, validation, and test loss, while ratio 1.0 had the highest sample BLEU but much worse loss. The very low BLEU scores and two-epoch duration make these provisional signals, not conclusive translation-quality rankings.

The two-epoch validation curves are saved as [`outputs/seq2seq_hidden_dimension_loss.png`](outputs/seq2seq_hidden_dimension_loss.png), [`outputs/seq2seq_teacher_forcing_loss.png`](outputs/seq2seq_teacher_forcing_loss.png), and [`outputs/seq2seq_sequence_length_loss.png`](outputs/seq2seq_sequence_length_loss.png). They provide a comparison of loss trends, not a reliable final ranking: each setting would need longer, preferably repeated, runs before drawing a strong conclusion.

## Q9 — Test evaluation

The original 10-epoch baseline (maximum length 20) had test loss **5.0915** and corpus BLEU **0.0550** on a quick-mode sample of 1,000 held-out pairs. After the short comparisons, I trained the selected length-10 setting for 10 epochs: hidden size 256, teacher forcing 0.5, and maximum length 10. This final model had test loss **4.7199** and corpus BLEU **0.0709** on 1,000 held-out pairs. The length-10 model truncated 1,429 English and 2,028 French training sentences, so its lower test loss and higher BLEU come with a substantial loss of sentence coverage; neither metric alone proves that it is the better practical model. BLEU compares generated text with reference translations, but it does not directly measure meaning or fluency and may penalize valid translations that use different wording.

The notebook includes the loss plot and translation examples. The model's main observed issues were omissions, repeated words, poor word choice, and grammar. Using only the final encoder states and a word-level vocabulary also limits its ability to handle longer sentences and unseen words.

## Q10 — What the results show

The original baseline used hidden size 256, teacher forcing 0.5, and maximum length 20. The lowest-validation-loss short comparison was the length-10 setting, so that configuration was trained for 10 epochs as a final selected run. It reached validation loss 4.7123, test loss 4.7199, and sample BLEU 0.0709. The baseline and selected model use different sequence lengths, and the selected model truncates many more sentences; this comparison is therefore a measured experiment, not evidence that length 10 is universally best.

## The 20 new English sentences

For this evaluation, 20 new English sentences with French references were generated with the user's permission. They include different sentence lengths and structures and are not copied directly from the translation dataset. The model's predictions and references are listed in [`outputs/original_sentence_evaluation.csv`](outputs/original_sentence_evaluation.csv). The references are intended for this experiment and may have translation variations.

## Limitations and conclusion

The baseline and selected model were trained on 12,000 of the available 175,621 pairs for 10 epochs on CPU. The nine comparisons used the same sample and split but only ran for two epochs; they are preliminary short-run evidence. The final selected checkpoint and its metrics are saved in [`outputs/seq2seq_final_selected.pth`](outputs/seq2seq_final_selected.pth) and [`outputs/seq2seq_final_summary.csv`](outputs/seq2seq_final_summary.csv); its training history and loss plot are in [`outputs/seq2seq_final_history.csv`](outputs/seq2seq_final_history.csv) and [`outputs/seq2seq_final_loss.png`](outputs/seq2seq_final_loss.png). The model uses no attention or subword tokens, and a 10-token maximum truncates many sentences. The new-prompt set is small, and its reference translations have not been independently reviewed.

The selected model improves the reported test loss and sample BLEU over the original baseline on the same held-out split, but both scores remain modest and its shorter length discards substantially more training text. The results show a trade-off rather than a definitive best configuration; longer and repeated runs, with attention to the truncation cost, would be needed to make that decision confidently.

## Direct answers to the assignment questions

### Q1 — Data loading and preprocessing

- **Why tokenize?** Tokenization turns sentences into units the model can represent and map to numerical IDs.
- **Why separate English and French vocabularies?** The languages have different words and token-to-ID mappings; separate vocabularies let the encoder and decoder use the correct IDs.
- **Why SOS and EOS?** SOS tells the decoder where generation starts; EOS marks the end of a translation.
- **Why pad?** Padding makes examples in a batch the same length so they can be processed together. I excluded PAD positions from the loss.
- **What are truncation's limitations?** It discards sentence content. My length-10 setting truncated 1,429 English and 2,028 French training sentences, which can remove context or even the fact needed for a correct translation.

### Q2 — Encoder

- **Hidden state:** It carries the encoder's current representation of the input up to that step.
- **Cell state:** It is the LSTM's gated memory path, which helps preserve information across time steps.
- **Why an LSTM over a vanilla RNN for longer sequences?** Its gates provide a path for retaining or forgetting information and help mitigate vanishing gradients; this can make learning long-range dependencies easier, though it does not guarantee they will be learned.
- **Final-state limitation:** My decoder receives only the final hidden and cell states, so the encoder must compress the whole input into them. Without attention, the decoder cannot revisit individual source words during generation.

### Q3 — Decoder

- **Why a separate French vocabulary?** The decoder predicts French tokens, which have different words and IDs from the English input vocabulary.
- **Why initialize from encoder states?** This conditions French generation on the encoded English sentence.
- **How is output length variable?** I start decoding with SOS and stop when EOS is generated or the maximum length is reached.
- **What does the output layer do?** It maps each decoder hidden state to scores (logits) over the French vocabulary; greedy decoding chooses the highest-scoring token.

### Q4 — Training

- **Teacher forcing:** During training, I sometimes feed the correct previous French token rather than the model's previous prediction; my baseline ratio was 0.5.
- **Why can it speed learning?** The decoder receives the correct prefix while learning each next-token prediction, instead of having early mistakes distort all later steps.
- **What happens at inference?** Inference uses the model's own previous outputs. The difference from training is exposure bias, and a wrong token can lead to further errors.
- **Why ignore PAD in loss?** PAD is not a real translation target; counting it would reward predicting padding rather than learning French.
- **How do losses reveal overfitting?** If training loss keeps falling while validation loss stops improving or rises, the model may be fitting the training set rather than generalizing. My baseline ended with training loss 3.6276 and validation loss 5.1018, a notable gap; the final length-10 run also had a gap (3.1903 training versus 4.7123 validation at epoch 10).

### Q5 — Greedy decoding

- **Short versus long sentences:** My held-out examples show errors even on short inputs (for example, “take a deep breath” was not translated correctly). A longer example, “I think we both know why I'm here,” also had missing/repeated words. Longer inputs can be harder because errors accumulate and the final-state encoder has a fixed-size representation; I did not calculate a separate length-stratified score.
- **Unknown words:** An English word outside the training vocabulary maps to `<UNK>`. The decoder's text conversion omits special tokens, so this unknown marker is not rendered as an ordinary French word.
- **Greedy-decoding limitation:** It chooses the locally highest-scoring token at each step and does not compare alternative complete sequences; an early mistake can constrain later output.

### Q6 — Hidden dimension

Increasing hidden size from 128 to 256 to 512 increased parameters from 2,656,892 to 4,083,836 to 7,724,156 and increased two-epoch training time from 63.81 to 96.46 to 209.74 seconds. Hidden size 128 had the best validation and test loss; sample BLEU was nearly tied for 128 and 256, then lower for 512. The larger models did not improve the measured results in these short runs. More parameters can raise overfitting risk, but these runs alone do not prove that the larger models overfit. I would choose 128 for the best measured validation-loss/computation trade-off, while treating the small BLEU differences cautiously.

### Q7 — Teacher-forcing ratio

The ratio-0.0 run had the lowest train, validation, and test loss (5.5555, 5.8423, and 5.8514). The ratio-1.0 run had the highest loss (6.4526, 6.6925, and 6.6974) but the highest sample BLEU in this group (0.0074 versus 0.0017 and 0.0019). Thus, the loss and small-sample BLEU disagree: I cannot claim a definitive best translation ratio. Higher teacher forcing can make training inputs differ more from inference inputs, increasing exposure bias; the 0.5 setting is a middle-ground baseline, not a proven winner here.

### Q8 — Maximum sequence length

Length 10 had the lowest two-epoch validation/test loss, the highest BLEU in this group, and the shortest training time, but truncated 1,429 English and 2,028 French training sentences. Length 20 truncated only 18 and 42 and took 80.42 seconds; length 30 truncated none but took 158.39 seconds and had the worst validation/test loss of these runs. All sentences remained in the sample, but truncation removed tokens from some of them. I selected length 10 based on validation results and then trained it for 10 epochs, while explicitly noting its coverage trade-off. Increasing length did not improve these measured scores.

### Q9 — Performance evaluation

My final selected model had test loss 4.7199 and corpus BLEU 0.0709 on 1,000 held-out test pairs. I saved ten held-out English/reference/prediction examples and the loss curve. The examples include omissions, repetitions, incorrect word choices, and grammar errors; both short and longer inputs can fail. The model has no attention, so its fixed final encoder states bottleneck source information. BLEU is useful for comparing n-gram overlap with references, but it can penalize valid paraphrases and does not directly measure meaning, adequacy, or fluency.

### Q10 — Experimental comparison and final selection

I chose the length-10 setting because it had the lowest validation loss in the nine two-epoch comparisons, then trained that configuration for 10 epochs. It produced test loss 4.7199 and BLEU 0.0709 on 1,000 examples, better than the original length-20 baseline's test loss 5.0915 and BLEU 0.0550 on the same quick-mode test sample. However, the comparison runs were only two epochs, use BLEU samples of up to 300, and did not retain separate translation examples for every trial; the saved held-out examples show the final model's errors, not a controlled qualitative comparison of all nine short runs.

- **Best measured translation setting?** Length 10 won on validation/test loss and BLEU among the sequence-length runs, and was selected for the final model. It is not a universal winner because it truncates substantially more text.
- **Effect of hidden size?** The 128-unit model used the fewest parameters and had the lowest validation/test loss; 256 had almost identical BLEU, while 512 took over three times as long as 128 without improving the measured scores.
- **Best teacher-forcing ratio?** Ratio 0.0 won on loss; ratio 1.0 had the highest very-low sample BLEU. The results do not identify a reliable overall winner.
- **Best maximum length?** Length 10 won on measured losses and BLEU, but length 20 preserves far more training text. I selected 10 from validation loss and documented the trade-off.
- **Long sentences?** They are more vulnerable to truncation, compression into final states, and accumulated decoding errors. My examples also show that shorter sentences can be mistranslated.
- **Final-model limitations?** Modest BLEU, frequent translation errors, no attention or subword modeling, unknown-word handling through `<UNK>`, greedy decoding, limited training sample, and truncation at length 10.

The nine-row metrics, epoch histories, and three comparison loss plots are included in `outputs/`. I drafted the 20 extra English/French sentence pairs with the user's permission; the model predictions are saved in `outputs/original_sentence_evaluation.csv`.

## References

1. Devicharith, *Language Translation English-French Dataset*, Kaggle. https://www.kaggle.com/datasets/devicharith/language-translation-englishfrench-dataset
2. Sutskever, I., Vinyals, O., and Le, Q. V. (2014). “Sequence to Sequence Learning with Neural Networks.” *Advances in Neural Information Processing Systems*.
3. Hochreiter, S., and Schmidhuber, J. (1997). “Long Short-Term Memory.” *Neural Computation*, 9(8), 1735–1780.
4. Papineni, K. et al. (2002). “BLEU: a Method for Automatic Evaluation of Machine Translation.” *ACL*.
