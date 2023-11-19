# -*- coding:utf-8 -*-
# File       : text_classify.py
# Time       : 22/8/2023 下午 11:55
# Author     ：rain
# Description：DNN 类型的模型训练
import os
import time

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from tools.progressbar import ProgressBar
from tools.common import seed_everything, init_logger, logger, plot_img_acc_loss
from models import TextCNN, TextBiLSTM, FastText
from processors.text_classify import convert_examples_to_features, convert_fasttext_features
from processors.text_classify import cls_processors as processors
from processors.text_classify import collate_fn, fasttext_collate_fn
from tools.finetuning_argparse import get_argparse
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def train_evaluate_test(args, train_dataset, dev_dataset, test_dataset, model):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    if args.model_type == "fasttext":
        collate_function = fasttext_collate_fn
    else:
        collate_function = collate_fn
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_function)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    args.warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size
                * args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
                )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    steps_trained_in_current_epoch = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    seed_everything(args.seed)
    dev_best_acc = 0
    early_stop = 0  # 防止过拟合, 及时停止训练
    train_loss_list, train_acc_list = [], []
    dev_loss_list, dev_acc_list = [], []
    tmp_train_loss_list, tmp_train_acc_list = [], []
    for _ in range(int(args.num_train_epochs)):
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        for step, batch in enumerate(train_dataloader):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "input_mask": batch[1],
                      "target": batch[2]}
            if args.model_type == "fasttext":
                # XLM and RoBERTa don"t use segment_ids
                inputs["gram2_ids"] = batch[4]
                inputs["gram3_ids"] = batch[5]
            # model outputs are always tuple in pytorch-transformers (see doc)
            outputs = model(**inputs)
            loss, logits = outputs
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            pbar(step, {'loss': loss.item()})
            tr_loss += loss.item()
            predict_all = torch.max(logits, 1)[1].cpu().numpy()
            train_acc = accuracy_score(batch[2].cpu(), predict_all)
            tmp_train_acc_list.append(train_acc)
            tmp_train_loss_list.append(loss.item())
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    print(" ")
                    if args.local_rank == -1:
                        # Only evaluate when single GPU otherwise metrics may not average well
                        dev_acc, dev_loss = evaluate(args, model, dev_dataset)
                        train_loss = np.mean(np.array(tmp_train_loss_list))
                        train_acc = np.mean(np.array(tmp_train_acc_list))
                        train_loss_list.append(train_loss)
                        train_acc_list.append(train_acc)
                        dev_loss_list.append(dev_loss)
                        dev_acc_list.append(dev_acc)
                        msg = ('Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  '
                               'Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}')
                        print(msg.format(global_step, loss.item(), train_acc, dev_loss, dev_acc))
                        tmp_train_loss_list, tmp_train_acc_list = [], []
                        if dev_acc > dev_best_acc:
                            dev_best_acc = dev_acc
                            save_path = os.path.join(args.output_dir, f"{args.model_type}.ckpt")
                            torch.save(model.state_dict(), save_path)
                            logger.info("Saving model to %s", save_path)
        logger.info("\n")
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
    plot_img_acc_loss(train_loss_list, dev_loss_list, "Loss", args.model_type)
    plot_img_acc_loss(train_acc_list, dev_acc_list, "Accuracy", args.model_type)
    test(args, model, test_dataset)


def evaluate(args, model, eval_dataset, flag=False):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    if args.model_type == "fasttext":
        collate_function = fasttext_collate_fn
    else:
        collate_function = collate_fn
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 collate_fn=collate_function)
    # Eval!
    logger.info("***** Running evaluation %s *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
    if isinstance(model, nn.DataParallel):
        model = model.module
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    for step, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "input_mask": batch[1],
                      "target": batch[2]}
            if args.model_type == "fasttext":
                # XLM and RoBERTa don"t use segment_ids
                inputs["gram2_ids"] = batch[4]
                inputs["gram3_ids"] = batch[5]
            # model outputs are always tuple in pytorch-transformers (see doc)
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            batch_preds = torch.max(logits, 1)[1].cpu()
            labels_all = np.append(labels_all, batch[2].cpu())
            predict_all = np.append(predict_all, batch_preds)
        if args.n_gpu > 1:
            tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        pbar(step)
    dev_acc = accuracy_score(labels_all, predict_all)
    dev_loss = eval_loss / nb_eval_steps
    if flag:
        report = classification_report(labels_all, predict_all, target_names=args.label_list, digits=4)
        confusion = confusion_matrix(labels_all, predict_all)
        return dev_acc, dev_loss, report, confusion
    return dev_acc, dev_loss


def test(args, model, dev_dataset):
    # test
    save_path = os.path.join(args.output_dir, f"{args.model_type}.ckpt")
    model.load_state_dict(torch.load(save_path))
    model.eval()
    test_acc, test_loss, test_report, test_confusion = evaluate(args, model, dev_dataset, flag=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)


def load_and_cache_examples(args, processor, data_type='train'):
    task = args.task_name
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()
    if data_type == 'train':
        max_length = args.train_max_seq_length
    else:
        max_length = args.eval_max_seq_length
    # 加载数据并保存为缓存文件
    cached_features_file = os.path.join(args.data_dir, 'cached_-{}_{}_{}_{}'.format(
        data_type,
        args.model_type,
        str(max_length),
        str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if data_type == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir)
        if args.model_type == 'fasttext':
            features = convert_fasttext_features(
                examples=examples, label2id=args.label2id,
                max_seq_length=max_length, vocab_dict=processor.vocab_dict,
                gram2_dict=processor.gram2_dict, gram3_dict=processor.gram3_dict
            )
        else:
            features = convert_examples_to_features(
                examples=examples, label2id=args.label2id,
                max_seq_length=max_length, vocab_dict=processor.vocab_dict
            )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_idx for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    if args.model_type == "fasttext":
        all_gram2_ids = torch.tensor([f.gram2_ids for f in features], dtype=torch.long)
        all_gram3_ids = torch.tensor([f.gram3_ids for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, 
                                all_lens, all_label_ids, all_gram2_ids, all_gram3_ids)
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_lens, all_label_ids)
    return dataset


def main():
    args = get_argparse().parse_args()    # 训练输入参数处理, 需要新增/修改参数可以进入get_argparse配置
    # 模型保存/日志
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = args.output_dir + '/{}'.format(args.model_type)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    time_ = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    init_logger(log_file=args.output_dir + f'/{args.model_type}-{args.task_name}-{time_}.log')
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    # 设置debug
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    # 设置cuda, gpu 并行
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16, )
    # Set seed
    seed_everything(args.seed)
    # emotion 任务, 用任务名确认数据处理器
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name](args.data_dir, word_type=args.word_type,
                                           data_format=args.data_format)
    args.label_list, args.label2id, args.id2label = processor.get_labels()
    num_labels = len(args.label2id)
    vocab_size = len(processor.vocab_dict)
    gram2_size = len(processor.gram2_dict)
    gram3_size = len(processor.gram3_dict)

    # 如果是多卡并行, 串行处理
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    # 加载预训练任务
    args.model_type = args.model_type.lower()
    # 由模型类别判断加载不同的模型

    train_dataset = load_and_cache_examples(args, processor, data_type='train')
    eval_dataset = load_and_cache_examples(args, processor, data_type='eval')
    test_dataset = load_and_cache_examples(args, processor, data_type='test')

    if args.model_type == 'cnn':
        weight = torch.Tensor([0.88, 0.12]).to(args.device)
        model = TextCNN(vocab_size=vocab_size, embedding_size=256, hidden_size=256,
                        loss_type=args.loss_type, num_classes=num_labels, weight=weight)
    elif args.model_type == "lstm":
        model = TextBiLSTM(vocab_size=vocab_size, embedding_size=256, loss_type=args.loss_type,
                           hidden_size=256, num_classes=num_labels)
    elif args.model_type == "lstm-attention":
        model = TextBiLSTM(vocab_size=vocab_size, embedding_size=256, loss_type=args.loss_type,
                           hidden_size=256, num_classes=num_labels, attention="add")
    elif args.model_type == "fasttext":
        model = FastText(vocab_size=vocab_size, gram2_size=gram2_size, gram3_size=gram3_size,
                         embedding_size=200, hidden_size=200, num_classes=num_labels)
    else:
        print("model type error...")
        return
    # 第0张卡运行完再继续, 防止有需要下载的, 等第一张卡下载完再运行其他的卡, 不用重复下载
    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    # Training & Evaluate & Test/Predict
    if args.do_train:
        train_evaluate_test(args, train_dataset, eval_dataset, test_dataset, model)
        

if __name__ == "__main__":
    main()
