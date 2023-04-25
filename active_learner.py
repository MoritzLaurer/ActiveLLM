
import numpy as np
import pandas as pd
import torch
import datasets
import copy
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import gc

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoConfig, AutoModelForNextSentencePrediction,
    TrainingArguments, Trainer, 
    T5ForConditionalGeneration, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer, GenerationConfig,
    AutoModelForSeq2SeqLM
)




class ActiveLearner:
    def __init__(self, seed=42):
        self.seed = seed
        self.n_iteration = 0
        #self.results_test = {}
        self.metrics = {}
        #self.predictions_corpus = {}
        self.index_al_sample = []
        self.index_train_all = []
        self.df_al_sample_per_iter = {}


    def load_pd_dataset(self, df_corpus=None, text_column="text", label_column="label", df_test=None, separate_testset=True):
        self.separate_testset = separate_testset
        
        # these dfs can be used for training and inference by NLI or not NLI
        # for non-NLI, they just stay like this, for NLI they are formatted downstream
        self.df_corpus_format = df_corpus#.reset_index(drop=True)
        if separate_testset:
            self.df_test_format = df_test#.reset_index(drop=True)

        # creating these dfs for records and to be able to sample training data from corpus
        # and to be able to reformat it differently than test set for NLI
        self.df_corpus_original = df_corpus#.reset_index(drop=True)
        self.df_corpus_original.index.name = "idx"
        if separate_testset:
            self.df_test_original = df_test#.reset_index(drop=True)
            self.df_test_original.index.name = "idx"


        # this df is never NLI formatted and is gradually updated like dataset["corpus"]. Necessary for AL sample functions
        #self.df_corpus_original_update = df_corpus.reset_index(drop=True)
        self.text_column = text_column
        self.label_column = label_column

    
    """def format_pd_dataset_for_generative(self, instruction=None, text_column="text"):
        #self.instruction = instruction
        self.df_corpus_original.index.name = "idx"
        #self.df_corpus_format[text_column] = instruction + "\n\n" + self.df_corpus_original[text_column]
        if self.separate_testset:
            self.df_test_original.index.name = "idx"
        #    self.df_test_format[text_column] = instruction + "\n\n" + self.df_test_original[text_column]
    """

    def format_pd_dataset_for_nli(self, hypo_label_dic=None):
        # only run for NLI
        self.hypo_label_dic = hypo_label_dic
        # only formatting to nli test format (not train format), because not training data yet, first needs to be sampled after first 0-shot run.
        self.df_corpus_format = self.format_nli_testset(df_test=self.df_corpus_original)
        if self.separate_testset:
            self.df_test_format = self.format_nli_testset(df_test=self.df_test_original)

    """def format_pd_trainset_for_nli(self, hypo_label_dic=None):
        # only run for NLI
        self.hypo_label_dic = hypo_label_dic
        self.df_corpus_format = self.format_nli_testset(df_test=self.df_corpus_original)

    def format_pd_testset_for_nli(self, hypo_label_dic=None):
        # only run for NLI
        self.hypo_label_dic = hypo_label_dic
        self.df_test_format = self.format_nli_testset(df_test=self.df_test_original)"""


    def format_nli_trainset(self, df_train=None):  # df_train=None, hypo_label_dic=None, random_seed=42
        print(f"\nFor NLI: Augmenting data by adding random not_entail examples to the train set from other classes within the train set.")
        print(f"Length of df_train before this step is: {len(df_train)}.")
        print(f"Max augmentation can be: len(df_train) * 2 = {len(df_train ) *2}. Can also be lower, if there are more entail examples than not-entail for a majority class")

        df_train_copy = df_train.copy(deep=True)
        df_train_lst = []
        for label_text, hypothesis in self.hypo_label_dic.items():
            ## entailment
            df_train_step = df_train_copy[df_train_copy.label_text == label_text].copy(deep=True)
            df_train_step["hypothesis"] = [hypothesis] * len(df_train_step)
            df_train_step["label"] = [0] * len(df_train_step)
            ## not_entailment
            df_train_step_not_entail = df_train_copy[df_train_copy.label_text != label_text].copy(deep=True)
            # could try weighing the sample texts for not_entail here. e.g. to get same n texts for each label
            df_train_step_not_entail = df_train_step_not_entail.sample(n=min(len(df_train_step), len(df_train_step_not_entail)), random_state=self.seed)  # can try sampling more not_entail here
            df_train_step_not_entail["hypothesis"] = [hypothesis] * len(df_train_step_not_entail)
            df_train_step_not_entail["label"] = [1] * len(df_train_step_not_entail)
            # append
            df_train_lst.append(pd.concat([df_train_step, df_train_step_not_entail]))
        df_train_copy = pd.concat(df_train_lst)

        # shuffle
        #df_train_copy = df_train_copy.sample(frac=1, random_state=self.seed)
        df_train_copy["label"] = df_train_copy.label.apply(int)
        print(f"For NLI:  not_entail training examples were added, which leads to an augmented training dataset of length {len(df_train_copy)}.\n")

        df_train_copy.index.name = "idx"

        #self.df_train = df_train_copy
        return df_train_copy


    def format_nli_testset(self, df_test=None):  # hypo_label_dic=None, df_test=None
        ## explode test dataset for N hypotheses
        # hypotheses
        hypothesis_lst = [value for key, value in self.hypo_label_dic.items()]
        print("Number of hypotheses/classes: ", len(hypothesis_lst), "\n")

        # label lists with 0 at alphabetical position of their true hypo, 1 for other hypos
        label_text_label_dic_explode = {}
        for key, value in self.hypo_label_dic.items():
            label_lst = [0 if value == hypo else 1 for hypo in hypothesis_lst]
            label_text_label_dic_explode[key] = label_lst

        df_test_copy = df_test.copy(deep=True)  # did this change the global df?
        df_test_copy["label"] = df_test_copy.label_text.map(label_text_label_dic_explode)
        df_test_copy["hypothesis"] = [hypothesis_lst] * len(df_test_copy)
        print(f"For normal test, N classifications necessary: {len(df_test_copy)}")

        # explode dataset to have K-1 additional rows with not_entail label and K-1 other hypotheses
        # ! after exploding, cannot sample anymore, because distorts the order to true label values, which needs to be preserved for evaluation multilingual-repo
        df_test_copy = df_test_copy.explode(["hypothesis", "label"])  # multi-column explode requires pd.__version__ >= '1.3.0'
        print(f"For NLI test, N classifications necessary: {len(df_test_copy)}\n")

        df_test_copy.index.name = "idx"

        return df_test_copy


    def load_model_tokenizer(self, model_name=None, method=None, label_text_alphabetical=None, model_max_length=256, 
                            config_params=None, model_params=None):
        self.model_params = model_params
        self.config_params = config_params
        
        if method == "nli":
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, model_max_length=model_max_length);
            model = AutoModelForSequenceClassification.from_pretrained(model_name);
        elif method == "nsp":
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, model_max_length=model_max_length);
            model = AutoModelForNextSentencePrediction.from_pretrained(model_name);
        elif method == "standard_dl":
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, model_max_length=model_max_length);
            # define config. label text to label id in alphabetical order
            label2id = dict(zip(np.sort(label_text_alphabetical), np.sort(pd.factorize(label_text_alphabetical, sort=True)[0]).tolist()))  # .astype(int).tolist()
            id2label = dict(zip(np.sort(pd.factorize(label_text_alphabetical, sort=True)[0]).tolist(), np.sort(label_text_alphabetical)))
            config = AutoConfig.from_pretrained(model_name, label2id=label2id, id2label=id2label);
            # load model with config
            model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config);
        elif method == "generative":
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, model_max_length=model_max_length);
            
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                **self.model_params
            );
            generation_config = GenerationConfig.from_pretrained(model_name, **self.config_params)
            model.generation_config = generation_config
        else:
            raise Exception(f"Method {method} not implemented.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        model.to(device);

        self.method = method
        self.model_max_length = model_max_length
        self.label_text_alphabetical = label_text_alphabetical
        self.model = model
        self.model_name = model_name
        self.tokenizer = tokenizer
        #return model, tokenizer


    def tokenize_hf_dataset(self):
        # tokenize corpus and test set
        dataset_corpus = datasets.Dataset.from_pandas(self.df_corpus_format, preserve_index=True)
        if self.separate_testset:
            dataset_test = datasets.Dataset.from_pandas(self.df_test_format, preserve_index=True)

        def tokenize_func_nli(examples):
            return self.tokenizer(examples[self.text_column], examples["hypothesis"], truncation=True, max_length=self.model_max_length)
        def tokenize_func_mono(examples):
            return self.tokenizer(examples[self.text_column], truncation=True, max_length=self.model_max_length)

        if self.method == "nli" or self.method == "nsp":
            dataset_corpus = dataset_corpus.map(tokenize_func_nli, batched=True)
            if self.separate_testset:
                dataset_test = dataset_test.map(tokenize_func_nli, batched=True)
        elif self.method == "standard_dl":
            dataset_corpus = dataset_corpus.map(tokenize_func_mono, batched=True)
            if self.separate_testset:
                dataset_test = dataset_test.map(tokenize_func_mono, batched=True)
        elif self.method == "generative":
            max_target_length = 16
            max_input_length = self.model_max_length - max_target_length
            
            # to deal with fact that labels are also input text / count towards max token limit
            def tokenize_func_generative(examples):
                model_inputs = self.tokenizer(
                    examples[self.text_column],
                    max_length=max_input_length,
                    truncation=True,
                    return_tensors="pt", padding=True
                )
                labels = self.tokenizer(
                    examples[self.label_column], max_length=max_target_length, truncation=True, return_tensors="pt", padding=True
                )
                model_inputs["labels"] = labels["input_ids"]
                return model_inputs
            
            dataset_corpus = dataset_corpus.map(tokenize_func_generative, batched=True)
            if self.separate_testset:
                dataset_test = dataset_test.map(tokenize_func_generative, batched=True)
        
        # trainer/model does not accept other columns
        if self.method == "generative":
            dataset_corpus = dataset_corpus.remove_columns(
                self.df_corpus_format.columns
            )
        
        dataset = datasets.DatasetDict(
            {"corpus": dataset_corpus,
             "train": None,  # because no trainset in the beginning. first needs to be sampled with al
             "test": dataset_test if self.separate_testset else None 
             }
        )

        self.dataset = dataset


    def set_train_args(self, hyperparams_dic=None, training_directory=None, disable_tqdm=False, **kwargs):
        # https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments
        self.hyperparams_dic = hyperparams_dic
        
        if self.method != "generative":
            train_args = TrainingArguments(
                output_dir=f"./{training_directory}",  # f'./{training_directory}',  #f'./results/{training_directory}',
                logging_dir=f"./{training_directory}/logs",  # f'./{training_directory}',  #f'./logs/{training_directory}',
                **hyperparams_dic,
                **kwargs,
                # num_train_epochs=4,
                # learning_rate=1e-5,
                # per_device_train_batch_size=8,
                # per_device_eval_batch_size=8,
                # warmup_steps=0,  # 1000, 0
                # warmup_ratio=0,  #0.1, 0.06, 0
                # weight_decay=0,  #0.1, 0
                # load_best_model_at_end=True,
                # metric_for_best_model="f1_macro",
                # fp16=True,
                # fp16_full_eval=True,
                # evaluation_strategy="no",  # "epoch"
                # seed=42,
                # eval_steps=300  # evaluate after n steps if evaluation_strategy!='steps'. defaults to logging_steps
                save_strategy="no",  # options: "no"/"steps"/"epoch"
                # save_steps=1_000_000,              # Number of updates steps before two checkpoint saves.
                save_total_limit=10,  # If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in output_dir
                logging_strategy="epoch",
                report_to="all",  # "all"
                disable_tqdm=disable_tqdm,
                # push_to_hub=False,
                # push_to_hub_model_id=f"{model_name}-finetuned-{task}",
            )
        else:
            train_args = Seq2SeqTrainingArguments(
                output_dir=f"./{training_directory}",  # f'./{training_directory}',  #f'./results/{training_directory}',
                logging_dir=f"./{training_directory}/logs",  # f'./{training_directory}',  #f'./logs/{training_directory}',
                **hyperparams_dic,
                **kwargs,
                #predict_with_generate=False,
                # num_train_epochs=4,
                # learning_rate=1e-5,
                # per_device_train_batch_size=8,
                # per_device_eval_batch_size=8,
                # warmup_steps=0,  # 1000, 0
                # warmup_ratio=0,  #0.1, 0.06, 0
                # weight_decay=0,  #0.1, 0
                # load_best_model_at_end=True,
                # metric_for_best_model="f1_macro",
                # fp16=True,
                # fp16_full_eval=True,
                # evaluation_strategy="no",  # "epoch"
                # seed=42,
                # eval_steps=300  # evaluate after n steps if evaluation_strategy!='steps'. defaults to logging_steps
                save_strategy="no",  # options: "no"/"steps"/"epoch"
                # save_steps=1_000_000,              # Number of updates steps before two checkpoint saves.
                save_total_limit=10,  # If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in output_dir
                logging_strategy="epoch",
                report_to="all",  # "all"
                disable_tqdm=disable_tqdm,
                # push_to_hub=False,
                # push_to_hub_model_id=f"{model_name}-finetuned-{task}",
            )
        # for n, v in best_run.hyperparameters.items():
        #    setattr(trainer.args, n, v)

        self.train_args = train_args
        #return train_args


    def train_test_infer(self):  #  dataset=None, model=None, tokenizer=None

        if self.method == "nli" or self.method == "nsp":
            compute_metrics = self.compute_metrics_nli_binary
        elif self.method == "standard_dl":
            compute_metrics = self.compute_metrics_standard
        elif self.method == "generative":
            # ! can probably be deleted
            compute_metrics = self.compute_metrics_generative
        else:
            raise Exception(f"Compute metrics for trainer not specified correctly: {self.method}")

        # if not first run, load new model to train again
        if self.dataset["train"] != None:
            self.load_model_tokenizer(model_name=self.model_name, method=self.method, label_text_alphabetical=self.label_text_alphabetical, model_max_length=self.model_max_length, config_params=self.config_params, model_params=self.model_params);

        # create trainer
        #label_text_alphabet = self.label_text_alphabet  # lambda does not seem to accept self.label... itself
        if self.method != "generative":
            trainer = Trainer(
                model=self.model,
                tokenizer=self.tokenizer,
                args=self.train_args,
                train_dataset=self.dataset["train"],  # ["train"].shard(index=1, num_shards=100),  # https://huggingface.co/docs/datasets/processing.html#sharding-the-dataset-shard
                eval_dataset=self.dataset["test"] if self.separate_testset == True else self.dataset["corpus"],  #self.dataset["test"],
                compute_metrics=lambda eval_pred: compute_metrics(eval_pred, label_text_alphabetical=self.label_text_alphabetical, only_return_probabilities=False)
            )
        else:
            data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
            
            trainer = Seq2SeqTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                args=self.train_args,
                train_dataset=self.dataset["train"],  # ["train"].shard(index=1, num_shards=100),  # https://huggingface.co/docs/datasets/processing.html#sharding-the-dataset-shard
                eval_dataset=self.dataset["test"] if self.separate_testset == True else self.dataset["corpus"],  #self.dataset["test"],
                compute_metrics=lambda eval_pred: compute_metrics(eval_pred, label_text_alphabetical=self.label_text_alphabetical, only_return_probabilities=False),
                data_collator=data_collator,
            )

        if self.dataset["train"] != None:
            trainer.train()
            self.model = trainer.model

        # code in case there is a separate test set
        #results_test = trainer.evaluate(eval_dataset=self.dataset["test"])  # eval_dataset=encoded_dataset["test"]
        #print("\n", results_test, "\n")
        #self.results_test.update({f"test_iter_{self.n_iteration}": results_test})
        
        # update trainer to only return NLI probabilities for analysis of corpus for AL
        # https://github.com/huggingface/transformers/blob/31d452c68b34c2567b62924ee0df40a83cbc52d5/src/transformers/trainer.py#L448
        #trainer.compute_metrics = lambda eval_pred: compute_metrics(eval_pred, label_text_alphabetical=self.label_text_alphabetical,
        #                                                            only_return_probabilities=True)

        ## inference on ('unlabeled') corpus for next sampling round
        if self.method != "generative":
            metrics = trainer.evaluate(eval_dataset=self.dataset["corpus"])  # eval_dataset=encoded_dataset["test"]
        elif self.method == "generative": 
            # need to calculate metrics and uncertainty outside of trainer
            # seq2seq trainer evaluate/prediction_step has issues like this hack https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/trainer_seq2seq.py#L273
            # this means that I cannot pass a generate_config, decode labels correctly, or calculate uncertainty in compute_metrics
            metrics = self.metrics_uncertainty_generative()
            
           
        #print(results_corpus["eval_hypo_probabilities_entail"])
        self.metrics.update({f"iter_{self.n_iteration}": metrics})
        #self.predictions_corpus = predictions_corpus

        self.n_iteration += 1
        self.clean_memory()

    
    def clean_memory(self):
        # del(model)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()


    def metrics_uncertainty_generative(self):
        import warnings
        from torch.utils.data import Dataset, DataLoader
        import tqdm
        self.clean_memory()
        
        # TODO implement metrics calculation with different test-set
        inputs = {key: torch.tensor(value, dtype=torch.long).to(self.model.device) for key, value in self.dataset["corpus"].remove_columns("idx").to_dict().items()}
        
        # batched inference to avoid memory issues
        class TokenizedTextDataset(Dataset):
            def __init__(self, tokenized_inputs):
                self.tokenized_inputs = tokenized_inputs
        
            def __len__(self):
                return len(self.tokenized_inputs["input_ids"])
        
            def __getitem__(self, idx):
                item = {key: value[idx] for key, value in self.tokenized_inputs.items()}
                return item
        
        dataset_inputs = TokenizedTextDataset(inputs)
        dataloader = DataLoader(dataset_inputs, batch_size=self.hyperparams_dic["per_device_eval_batch_size"], shuffle=False)
        
        reconstructed_scores = []
        labels_pred = []
        
        #with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Inference"):
            inputs_batched = {k: v.to(self.model.device) for k, v in batch.items()}
            outputs = self.model.generate(
                **inputs_batched,
                **{key: value for key, value in self.config_params.items() if key != "generation_num_beams"},
            )
            
            # compute transition scores for sequences differently if beamsearch
            if self.config_params["num_beams"] == 1:
                transition_scores = self.model.compute_transition_scores(
                    outputs.sequences, outputs.scores, normalize_logits=False, #outputs.beam_indices
                )
            else:
                transition_scores = self.model.compute_transition_scores(
                    outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False
                )

            ## get scores for entire sequence
            # If you sum the generated tokens' scores and apply the length penalty, you'll get the sequence scores.
            # Tip: set `normalize_logits=True` to recompute the scores from the normalized logits.
            output_length = inputs["input_ids"].shape[1] + np.sum(transition_scores.to(torch.float32).cpu().numpy() < 0, axis=1)
            length_penalty = self.model.generation_config.length_penalty
            reconstructed_scores_batch = transition_scores.to(torch.float32).cpu().sum(axis=1) / (output_length**length_penalty)
            reconstructed_scores.append(reconstructed_scores_batch.tolist())
        
            # get predicted label strings
            labels_pred_batch = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
            labels_pred.append(labels_pred_batch)
        
        reconstructed_scores = [item for sublist in reconstructed_scores for item in sublist]
        labels_pred = [item for sublist in labels_pred for item in sublist]
        
        
        # ! deletable memory inefficient generation
        """#inputs = self.tokenizer(self.dataset["corpus"][self.text_column], return_tensors="pt")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {key: torch.tensor(value, dtype=torch.long).to(device) for key, value in self.dataset["corpus"].remove_columns("idx").to_dict().items()}

        outputs = self.model.generate(
            **inputs,
            **self.config_params,
        )
        if self.config_params["num_beams"] == 1:
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=False, #outputs.beam_indices
            )
        else:
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False
            )
        #transition_scores = self.model.compute_transition_scores(
        #    outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False
        #)

        ## get scores for entire sequence
        # If you sum the generated tokens' scores and apply the length penalty, you'll get the sequence scores.
        # Tip: set `normalize_logits=True` to recompute the scores from the normalized logits.
        output_length = inputs["input_ids"].shape[1] + np.sum(transition_scores.to(torch.float32).cpu().numpy() < 0, axis=1)
        length_penalty = self.model.generation_config.length_penalty
        reconstructed_scores = transition_scores.to(torch.float32).cpu().sum(axis=1) / (output_length**length_penalty)
        
        # get predicted label strings
        labels_pred = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)"""


        # get true labels
        df_corpus_format_notrain = self.df_corpus_format[~self.df_corpus_format.index.isin(self.index_train_all)]
        labels_gold = df_corpus_format_notrain[self.label_column].tolist()
        
        #dataset_corpus_update = self.dataset["corpus"].filter(lambda example: example["idx"] not in self.index_train_all)

        
        ### calculate metrics
        warnings.filterwarnings('ignore')
        ## metrics
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels_gold, labels_pred, average='macro', zero_division=0)  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(labels_gold, labels_pred, average='micro', zero_division=0)  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
        acc_balanced = balanced_accuracy_score(labels_gold, labels_pred)
        acc_not_balanced = accuracy_score(labels_gold, labels_pred)
    
        metrics = {
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'accuracy_balanced': acc_balanced,
            #'accuracy_not_b': acc_not_balanced,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            #'label_gold_raw': labels_gold,
            #'label_predicted_raw': labels_pred
        }
        # rounding
        metrics = {key : round(metrics[key], 3) if key not in ["label_gold_raw", "label_predicted_raw"] else {key: metrics[key]} for key in metrics}
    
        print("  Aggregate metrics: ", {key: metrics[key] for key in metrics if key not in ["label_gold_raw", "label_predicted_raw"]} )  # print metrics but without label lists
        #print("Detailed metrics: ", classification_report(labels, preds_max, labels=np.sort(pd.factorize(label_text_alphabetical, sort=True)[0]), target_names=label_text_alphabetical, sample_weight=None, digits=2, output_dict=True, zero_division='warn'), "\n")
    
        # clean memory    
        self.clean_memory()
        
        
        self.iteration_label_gold = labels_gold
        self.iteration_label_predicted = labels_pred
        # also store the probabilities for the al sampling strategy
        self.iteration_probabilities = reconstructed_scores
        
        warnings.filterwarnings('default')
        return metrics


    def compute_metrics_standard(self, eval_pred, label_text_alphabetical=None):
        labels = eval_pred.label_ids
        pred_logits = eval_pred.predictions
        preds_max = np.argmax(pred_logits, axis=1)  # argmax on each row (axis=1) in the tensor
        #print(labels)
        #print(preds_max)
        ## metrics
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds_max,
                                                                                     average='macro')  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(labels, preds_max,
                                                                                     average='micro')  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
        acc_balanced = balanced_accuracy_score(labels, preds_max)
        acc_not_balanced = accuracy_score(labels, preds_max)

        metrics = {'f1_macro': f1_macro,
                   'f1_micro': f1_micro,
                   'accuracy_balanced': acc_balanced,
                   'accuracy_not_b': acc_not_balanced,
                   'precision_macro': precision_macro,
                   'recall_macro': recall_macro,
                   'precision_micro': precision_micro,
                   'recall_micro': recall_micro,
                   #'label_gold_raw': labels,
                   #'label_predicted_raw': preds_max
                   }
        print("Aggregate metrics: ", {key: metrics[key] for key in metrics if key not in ["label_gold_raw", "label_predicted_raw"]})  # print metrics but without label lists
        print("Detailed metrics: ",
              classification_report(labels, preds_max, labels=np.sort(pd.factorize(label_text_alphabetical, sort=True)[0]), target_names=label_text_alphabetical, sample_weight=None, digits=2,
                                    output_dict=True,
                                    zero_division='warn'), "\n")
        
        # store the respective label values and predictions for each iteration in order to be able to extract it later for downstream analyses             
        self.iteration_label_gold = labels
        self.iteration_label_predicted = preds_max
        # also store the probabilities for the al sampling strategy
        #self.iteration_probabilities = hypo_probabilities_entail
        
        return metrics


    def compute_metrics_nli_binary(self, eval_pred, label_text_alphabetical=None, only_return_probabilities=False):
        predictions, labels = eval_pred

        # split in chunks with predictions for each hypothesis for one unique premise
        def chunks(lst, n):  # Yield successive n-sized chunks from lst. https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        # for each chunk/premise, select the most likely hypothesis, either via raw logits, or softmax
        select_class_with_softmax = True  # tested this on two datasets - output is exactly (!) the same. makes no difference.
        softmax = torch.nn.Softmax(dim=1)
        prediction_chunks_lst = list(chunks(predictions, len(set(label_text_alphabetical))))  # len(LABEL_TEXT_ALPHABETICAL)
        hypo_position_highest_prob = []
        hypo_probabilities_entail = []
        for i, chunk in enumerate(prediction_chunks_lst):
            # if else makes no empirical difference. resulting metrics are exactly the same
            if select_class_with_softmax:
                chunk_tensor = torch.tensor(chunk, dtype=torch.float32)
                chunk_tensor = softmax(chunk_tensor)  # .tolist()
                hypo_position_highest_prob.append(np.argmax(np.array(chunk)[:, 0]))  # only accesses the first column of the array, i.e. the entailment prediction logit of all hypos and takes the highest one
                hypo_probabilities_entail.append(chunk_tensor[:, 0].tolist())  # only accesses the first column of the array, i.e. the entailment softmax score of all hypos and take all
            else:
                # argmax on raw logits
                hypo_position_highest_prob.append(np.argmax(chunk[:, 0]))  # only accesses the first column of the array, i.e. the entailment prediction logit of all hypos and takes the highest one

        #if not only_return_probabilities:
        label_chunks_lst = list(chunks(labels, len(set(label_text_alphabetical))))
        label_position_gold = []
        for chunk in label_chunks_lst:
            label_position_gold.append(np.argmin(chunk))  # argmin to detect the position of the 0 among the 1s

        ## metrics
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(label_position_gold, hypo_position_highest_prob, average='macro')  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(label_position_gold, hypo_position_highest_prob, average='micro')  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
        acc_balanced = balanced_accuracy_score(label_position_gold, hypo_position_highest_prob)
        acc_not_balanced = accuracy_score(label_position_gold, hypo_position_highest_prob)
        metrics = {'f1_macro': f1_macro,
                   'f1_micro': f1_micro,
                   'accuracy_balanced': acc_balanced,
                   'accuracy_not_b': acc_not_balanced,
                   'precision_macro': precision_macro,
                   'recall_macro': recall_macro,
                   'precision_micro': precision_micro,
                   'recall_micro': recall_micro,
                   #'label_gold_raw': label_position_gold,
                   #'label_predicted_raw': hypo_position_highest_prob
                   }
        print("Aggregate metrics: ", {key: metrics[key] for key in metrics if key not in ["label_gold_raw", "label_predicted_raw"]})  # print metrics but without label lists
        print("Detailed metrics: ",
              classification_report(label_position_gold, hypo_position_highest_prob, labels=np.sort(pd.factorize(label_text_alphabetical, sort=True)[0]), target_names=label_text_alphabetical,
                                    sample_weight=None, digits=2, output_dict=True,
                                    zero_division='warn'), "\n")
        
        # store the respective label values and predictions for each iteration in order to be able to extract it later for downstream analyses             
        self.iteration_label_gold = label_position_gold
        self.iteration_label_predicted = hypo_position_highest_prob
        # also store the probabilities for the al sampling strategy
        self.iteration_probabilities = hypo_probabilities_entail

        return metrics


    def compute_metrics_generative(self, eval_pred, label_text_alphabetical=None, only_return_probabilities=False):
        
        self.eval_pred = eval_pred

        ### trying to reuse older code from standard metrics
        import warnings     
        predictions, labels, inputs = eval_pred
        output_logits = predictions[0]  # e.g. (20, 6, 32128)
        input_embeddings = predictions[1]  # e.g. (20, 220, 1024)
        
        # get predicted token ids
        # do I need to offset this somehow?
        predictions_token_id = np.argmax(output_logits, axis=-1)
        # Replace -100 in the prediction as we can't decode them
        predictions_token_id = np.where(output_logits != -100, predictions_token_id, self.tokenizer.pad_token_id)
        # decode and lower prediction
        labels_pred = self.tokenizer.batch_decode(predictions_token_id, skip_special_tokens=True)
        #labels_pred = [label.lower() for label in labels_pred]
        
        # decode and lower gold label
        # Replace -100 in the labels as we can't decode them
        labels_gold = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        labels_gold = self.tokenizer.batch_decode(labels_gold, skip_special_tokens=True)
        #labels_gold = [label.lower() for label in labels_gold]
    
        warnings.filterwarnings('ignore')
        ## metrics
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels_gold, labels_pred, average='macro', zero_division=0)  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(labels_gold, labels_pred, average='micro', zero_division=0)  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
        acc_balanced = balanced_accuracy_score(labels_gold, labels_pred)
        acc_not_balanced = accuracy_score(labels_gold, labels_pred)
    
        metrics = {
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'accuracy_balanced': acc_balanced,
            #'accuracy_not_b': acc_not_balanced,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            #'label_gold_raw': labels_gold,
            #'label_predicted_raw': labels_pred
        }
        # rounding
        metrics = {key : round(metrics[key], 3) if key not in ["label_gold_raw", "label_predicted_raw"] else {key: metrics[key]} for key in metrics}
    
        print("  Aggregate metrics: ", {key: metrics[key] for key in metrics if key not in ["label_gold_raw", "label_predicted_raw"]} )  # print metrics but without label lists
        #print("Detailed metrics: ", classification_report(labels, preds_max, labels=np.sort(pd.factorize(label_text_alphabetical, sort=True)[0]), target_names=label_text_alphabetical, sample_weight=None, digits=2, output_dict=True, zero_division='warn'), "\n")
    
        # clean memory    
        #clean_memory()
        
        
        ## add calculation of probabilities for AL loop
        """
        ### re-calculation with full outputs since trainer does not return full outputs
        ## get decoded response
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=4,
            num_return_sequences=4,
            temperature=1,  # default: 1.0
            top_k=70,  # default: 50
            return_dict_in_generate=True,
            output_scores=True,
        )  
        ## get uncertainty score
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=True
        )
        # to cpu to enable numpy operations
        transition_scores = transition_scores.to(torch.float32).cpu().numpy()
        ## get scores for entire sequence
        # If you sum the generated tokens' scores and apply the length penalty, you'll get the sequence scores.
        # Tip: set `normalize_logits=True` to recompute the scores from the normalized logits.
        output_length = inputs.input_ids.shape[1] + np.sum(transition_scores < 0, axis=1)
        length_penalty = self.model.generation_config.length_penalty
        reconstructed_scores = transition_scores.sum(axis=1) / (output_length**length_penalty)"""

        
        # naive calculation of certainty by summing highest logist
        # Get the largest logit along the last dimension
        # no idea if should remove specific tokens or offset or sth; should apply length penalty etc. 
        max_logits = np.max(output_logits, axis=-1)
        probabilities_sequence = np.sum(max_logits, axis=1)

        
        self.iteration_label_gold = labels_gold
        self.iteration_label_predicted = labels_pred
        # also store the probabilities for the al sampling strategy
        self.iteration_probabilities = probabilities_sequence  #reconstructed_scores
        
        warnings.filterwarnings('default')
        return metrics


    ### custom query strategies
    def min_certainty(self, n_sample_al=20):
        probs_arr = np.array(self.iteration_probabilities)
        
        # Get indices of n smallest numbers
        probs_min_indices = np.argpartition(probs_arr, n_sample_al)[:n_sample_al]
        # Extract the n smallest numbers from the array
        probs_min = probs_arr[probs_min_indices]
        
        ## get df_corpus, without the rows already used for training
        df_corpus_original_without_train = self.df_corpus_original[~self.df_corpus_original.index.isin(self.index_train_all)].copy(deep=True)
        if len(df_corpus_original_without_train) == 0:
            raise Exception("No more training data left in corpus.")
        #df_corpus_ties = self.df_corpus_original_update.copy(deep=True)
        #df_corpus_ties = df_corpus_original_without_train
        df_corpus_original_without_train["label_pred_probs"] = probs_arr

        ## extract only those rows with the smallest class distance
        #df_corpus_ties_sample = df_corpus_ties[df_corpus_ties.index.isin(entail_distance_min.index)]
        df_corpus_sample = df_corpus_original_without_train.iloc[probs_min_indices]
        # shuffle for random training sequence instead of ordered by difficulty/ties/uncertainty
        df_corpus_sample = df_corpus_sample.sample(frac=1, random_state=self.seed)

        # scale probabilities to sum to 1 to enable argilla to ingest it
        #df_corpus_ties_sample["probs_entail_scaled"] = [dict(zip(list(probs_entail_dic.keys()), np.array(list(probs_entail_dic.values())) / np.array(list(probs_entail_dic.values())).sum())) for probs_entail_dic in df_corpus_ties_sample.probs_entail]

        index_al_sample = df_corpus_sample.index.tolist()
        
        self.index_al_sample = index_al_sample
        self.df_corpus_al_sample = df_corpus_sample
        self.df_al_sample_per_iter.update({f"iter_{self.n_iteration}": df_corpus_sample})
        # deletable, just for inspection
        self.df_corpus_with_probs = df_corpus_original_without_train

        
    
    def sample_breaking_ties(self, n_sample_al=20):
        hypo_prob_entail = self.iteration_probabilities
        # mapping entail probabilities to labels
        hypo_prob_entail = [{label_text: round(entail_score, 4) for entail_score, label_text in zip(prob_entail, self.label_text_alphabetical)} for prob_entail in hypo_prob_entail]

        ## get values and indexes from breaking ties sampling strategy
        entail_distance = [pd.Series(prob_entail.values()).nlargest(n=2).max() - pd.Series(prob_entail.values()).nlargest(n=2).min() for prob_entail in hypo_prob_entail]
        # select N hardest ties for active learning
        entail_distance_min = pd.Series(entail_distance).nsmallest(n=n_sample_al)
        # model prediction
        entail_max = [max(prob_entail, key=prob_entail.get) for prob_entail in hypo_prob_entail]

        ## get df_corpus, without the rows already used for training
        df_corpus_original_without_train = self.df_corpus_original[~self.df_corpus_original.index.isin(self.index_train_all)].copy(deep=True)
        if len(df_corpus_original_without_train) == 0:
            raise Exception("No more training data left in corpus.")
        #df_corpus_ties = self.df_corpus_original_update.copy(deep=True)
        #df_corpus_ties = df_corpus_original_without_train
        df_corpus_original_without_train["probs_entail"] = hypo_prob_entail
        df_corpus_original_without_train["label_text_pred"] = entail_max
        
        ## extract only those rows with the smallest class distance
        #df_corpus_ties_sample = df_corpus_ties[df_corpus_ties.index.isin(entail_distance_min.index)]
        df_corpus_ties_sample = df_corpus_original_without_train.iloc[entail_distance_min.index]
        # shuffle for random training sequence instead of ordered by difficulty/ties/uncertainty
        df_corpus_ties_sample = df_corpus_ties_sample.sample(frac=1, random_state=self.seed)

        # scale probabilities to sum to 1 to enable argilla to ingest it
        df_corpus_ties_sample["probs_entail_scaled"] = [dict(zip(list(probs_entail_dic.keys()), np.array(list(probs_entail_dic.values())) / np.array(list(probs_entail_dic.values())).sum())) for probs_entail_dic in df_corpus_ties_sample.probs_entail]

        index_al_sample = df_corpus_ties_sample.index.tolist()

        self.index_al_sample = index_al_sample
        self.df_corpus_al_sample = df_corpus_ties_sample
        self.df_al_sample_per_iter.update({f"iter_{self.n_iteration}": df_corpus_ties_sample})


    # remove sampled texts from corpus
    def update_dataset(self, label_annotation=None):
        #global dataset_update
        print("Examples in previous corpus iteration: ", len(set(self.dataset["corpus"]["idx"])))

        ## update index for entire train set, not only new sample
        self.index_train_all = list(set(self.index_train_all + self.index_al_sample))

        ## update df_corpus_original_update, because using it for sampling strategies
        """if len(self.df_corpus_original_update) == 0:
            raise Exception("No more training data left in corpus.")
        df_corpus_original_update = self.df_corpus_original_update.copy(deep=True)
        self.df_corpus_original_update = df_corpus_original_update[~df_corpus_original_update.index.isin(self.index_train_all)]"""

        ## create, resample and format train set to NLI train format
        # !! for argilla this has to always be on the entire corpus, because records accumulate 
        # !! and values of past annotations can also change if changing past annotations in interface
        # therefore I cannot just reuse self.df_corpus_al_sample, because annotation values might have changed in the meantime
        df_corpus_original = self.df_corpus_original.copy(deep=True)
        df_corpus_al_sample = df_corpus_original[df_corpus_original.index.isin(self.index_train_all)]
        #df_corpus_al_sample["label_from_dataset"] = df_corpus_al_sample["label"]
        #df_corpus_al_sample["label"] = label_annotation
        if self.method == "nli":
            # reformat training data for NLI training format
            df_train_update = self.format_nli_trainset(df_corpus_al_sample)
        elif self.method == "generative":
            df_train_update = df_corpus_al_sample
        else:
            raise Exception("Training data storage and formatting not implemented for non-NLI use-cases")

        ## tokenize trainset and format to hf
        # this needs to be repeated at every al iteration, because the new sampled training data needs to be converted to the NLI training format (above) and then tokenized
        dataset_train_update = datasets.Dataset.from_pandas(df_train_update, preserve_index=True)

        def tokenize_func_nli(examples):
            return self.tokenizer(examples[self.text_column], examples["hypothesis"], truncation=True, max_length=self.model_max_length)
        def tokenize_func_mono(examples):
            return self.tokenizer(examples[self.text_column], truncation=True, max_length=self.model_max_length)

        if self.method == "nli" or self.method == "nsp":
            dataset_train_update = dataset_train_update.map(tokenize_func_nli, batched=True)
        elif self.method == "standard_dl":
            dataset_train_update = dataset_train_update.map(tokenize_func_mono, batched=True)
        elif self.method == "generative":
            max_target_length = 16
            max_input_length = self.model_max_length - max_target_length

            # to deal with fact that labels are also input text / count towards max token limit
            def tokenize_func_generative(examples):
                model_inputs = self.tokenizer(
                    examples[self.text_column],
                    max_length=max_input_length,
                    truncation=True,
                    return_tensors="pt", padding=True
                )
                labels = self.tokenizer(
                    examples[self.label_column], max_length=max_target_length, truncation=True, return_tensors="pt", padding=True
                )
                model_inputs["labels"] = labels["input_ids"]
                return model_inputs
            
            dataset_train_update = dataset_train_update.map(tokenize_func_generative, batched=True)

        
        # give trainer only relevant columns
        if self.method == "generative":
            dataset_train_update = dataset_train_update.remove_columns(
                #dataset_eurepoc["train"].column_names
                self.df_corpus_format.columns
            )

        print("Number of new training data: ", len(set(dataset_train_update["idx"])))
        # not sure if this is necessary - guard against reruns with empty training dataset
        assert len(dataset_train_update["idx"]) > 0, "The dataset corpus has probably already been updated. If the dataset is updated again, no new training dataset would be created, as the training data index was already removed from the corpus"


        ## update hf dataset_corpus by excluding the training data
        dataset_corpus_update = self.dataset["corpus"].filter(lambda example: example["idx"] not in self.index_train_all)
        print("Examples in new corpus data without newly sampled training data: ", len(set(dataset_corpus_update["idx"])))


        self.dataset = datasets.DatasetDict(
            {"corpus": dataset_corpus_update,  # excludes training data
             "train": dataset_train_update.shuffle(seed=self.seed),
             "test": self.dataset["test"] if self.separate_testset == True else None  #dataset_corpus_update
             }
        )





